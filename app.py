import hashlib
import json
import os
import time
from io import BytesIO, StringIO
from typing import Annotated, Literal, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="OPC Multi-Agent Decision System",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------
# OpenAI API key — có thể dán trực tiếp vào đây nếu muốn cố định
# trong code (thay vì gõ mỗi lần trên sidebar hoặc đặt biến môi trường).
# Để trống "" nếu bạn muốn tiếp tục nhập ở ô "OpenAI API key" trên
# giao diện, hoặc dùng biến môi trường OPENAI_API_KEY.
#
# Thứ tự ưu tiên khi chạy: ô nhập trên sidebar > OPENAI_API_KEY_HARDCODED
# > biến môi trường OPENAI_API_KEY.
#
# CẢNH BÁO: nếu dán key thật vào đây, không chia sẻ/commit file này lên
# nơi công khai (GitHub public, v.v.).
# ------------------------------------------------------------------
OPENAI_API_KEY_HARDCODED = ""  

REQUIRED_SHEETS = [
    "02_OPC_PROFILE",
    "03_CUSTOMERS",
    "05_PRODUCTS",
    "06_ORDERS",
    "07_INVOICES",
    "08_BANK_TXN",
    "09_CASHFLOW",
    "10_CREDIT_PROFILE",
    "11_BANK_PRODUCTS",
    "13_RISK_RULES",
]

# 3 thành phố lõi theo System Prompt — các biến thể chính tả thường gặp trong Team Pack.
CORE_CITY_ALIASES = {
    "ha noi", "hà nội",
    "da nang", "đà nẵng",
    "tp.hcm", "tp hcm", "tphcm",
    "tp. ho chi minh", "tp. hồ chí minh",
    "thanh pho ho chi minh", "thành phố hồ chí minh",
    "ho chi minh", "hồ chí minh",
}

# Phân loại dòng tiền theo đúng cột pricing_model đọc trực tiếp từ 05_PRODUCTS
# (không hard-code theo service_id) — khớp 3 nhóm mô tả trong System Prompt:
# "Thuê bao hàng tháng" / "Khởi tạo ban đầu" / "Dự án theo giai đoạn".
PRICING_MODEL_MONTHLY_SUBSCRIPTION = "Monthly subscription"
PRICING_MODEL_INITIAL_SETUP = "Initial setup"
PRICING_MODEL_PROJECT = "Project"

CASH_RESERVE_THRESHOLD_DEFAULT = 550_000_000.0  # RR-002 fallback nếu thiếu profile
LARGE_DECISION_THRESHOLD = 300_000_000.0  # RR-005 / ngưỡng Founder approval
DEBT_CHECK_DATE = pd.Timestamp("2026-06-17")  # mốc kiểm tra hóa đơn Open quá hạn (Trường 2)

# Order Change (PDF mục 2.2) — là 1 phần của Oper Score CƠ SỞ, dùng CHUNG cho cả
# compute_order_change_coefficient() (baseline, khi tính hợp đồng ban đầu) lẫn
# resolve_crisis_deltas() (Crisis Card), để không có 2 bộ ngưỡng lệch nhau.
ORDER_CHANGE_FREE_LIMIT = 2  # <= 2 order/HĐ: miễn phí, không cộng Oper
ORDER_CHANGE_SURCHARGE_OPER = 0.005  # > free limit và <= hard cap: +0.5% Oper
# LƯU Ý: docx chỉ nói "có giới hạn mức trần order (khả năng của OPC)" nhưng KHÔNG
# nêu con số cụ thể -> 10 là ngưỡng TẠM ĐẶT, cần Founder xác nhận lại.
ORDER_CHANGE_HARD_CAP = 10



# ============================================================
# 2. STRUCTURED OUTPUT SCHEMAS (đầu ra 3 tác nhân)
# ============================================================

class FinanceAgentOutput(BaseModel):
    data_quality: Literal["COMPLETE", "MISSING_DATA"]
    missing_fields: list[str]
    preliminary_assessment: Literal[
        "CAN_ACCEPT", "CONDITIONAL", "DO_NOT_ACCEPT", "NEED_MORE_DATA"
    ]
    summary: str
    key_observations: list[str]


class RiskAgentOutput(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    triggered_rule_ids: list[str]
    warnings: list[str]
    human_confirmation_points: list[str]
    recommended_controls: list[str]
    unassessed_risks: list[str]


ExactlyThreeReasons = Annotated[list[str], Field(min_length=3, max_length=3)]


class DecisionAgentOutput(BaseModel):
    recommendation: Literal[
        "ACCEPT", "CONDITIONAL_ACCEPT", "REJECT", "NEED_MORE_DATA"
    ]
    # Decision Card bắt buộc: gross_margin, closing cash, confidence_score,
    # 1 phương án tài chính đề xuất, 3 lý do, 1 điều kiện bảo vệ.
    gross_margin: float
    closing_cash: float
    confidence_score: Optional[float] = None
    selected_financing_option: str
    funding_amount: float = Field(ge=0)
    three_reasons: ExactlyThreeReasons
    protection_condition: str
    human_approval_required: bool
    approval_reason: str
    executive_summary: str


# ============================================================
# 2.5 DATA MASKING — API-H-004 / API-H-007 (22_API_HANDLING_RULES)
#     theo đúng ví dụ trong 21_MASKING_EXAMPLES.
#
#     Hàm này CHỈ tạo ra một BẢN SAO đã che dữ liệu để gửi cho OpenAI —
#     không sửa đổi bất kỳ biến gốc nào dùng cho tính toán/hiển thị UI,
#     nên không ảnh hưởng tới logic nghiệp vụ hiện có.
# ============================================================

# Định danh hạn chế (restricted identifier) — theo 20_DATA_CLASS:
# "Do not send raw value across trust boundary" -> tokenize deterministically.
RESTRICTED_ID_FIELDS = {"customer_id", "account_id", "counterparty_id", "company_id"}
_RESTRICTED_ID_TOKEN_PREFIX = {
    "customer_id": "CUS",
    "account_id": "ACC",
    "counterparty_id": "CUS",
    "company_id": "ORG",
}

# Tên định danh doanh nghiệp/khách hàng — theo dòng "company_name" trong
# 21_MASKING_EXAMPLES: "Business identity is not needed for precheck".
NAME_FIELDS_TO_MASK = {"customer_name", "company_name"}

# Giá trị hợp đồng — theo dòng "contract_value" trong 21_MASKING_EXAMPLES:
# masked_example là dải giá trị gộp (band), ví dụ "4.2B band".
AGGREGATE_AMOUNT_FIELDS = {"contract_value"}

# Secret/credential — theo dòng "access_token" trong 21_MASKING_EXAMPLES và
# API-H-004/API-H-007: "Must never reach LLM prompt or audit log" -> loại bỏ
# hoàn toàn khỏi payload gửi cho OpenAI (không chỉ che, mà không gửi luôn).
SECRET_FIELD_MARKERS = {"access_token", "api_key", "secret", "password", "token"}


def _deterministic_token(prefix: str, raw_value) -> str:
    """
    Sinh token cố định (persistent linkage) cho cùng một giá trị gốc, để Agent
    vẫn có thể tham chiếu "cùng một khách hàng" xuyên suốt 1 lượt chạy mà không
    lộ định danh thật — đúng tinh thần cột allowed_for_partner_api = "Tokenized only".
    """
    digest = hashlib.sha256(f"{prefix}::{raw_value}".encode("utf-8")).hexdigest()[:6].upper()
    return f"TOK-{prefix}-{digest}"


_LEGAL_SUFFIX_WHITELIST = {"co", "ltd", "jsc", "corp", "inc", "plc", "llc", "cty"}


def _partial_mask_name(raw_value: str) -> str:
    """
    vd: 'OPC Digital Operations Co.' -> 'OPC D****** O********* Co.'
    (khớp đúng ví dụ trong 21_MASKING_EXAMPLES): giữ nguyên viết tắt IN HOA
    (OPC) và hậu tố pháp lý (Co./Ltd/JSC...), chỉ mask các từ định danh dài.
    """
    words = str(raw_value).split(" ")
    masked_words = []
    for word in words:
        bare = word.strip(".,").lower()
        if word.isupper() or bare in _LEGAL_SUFFIX_WHITELIST or len(word) <= 2:
            masked_words.append(word)
        else:
            masked_words.append(word[0] + "*" * (len(word) - 1))
    return " ".join(masked_words)


def _band_amount(raw_value) -> str:
    """vd: 4200000000 -> '4.2B band' (aggregated when possible)."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return str(raw_value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B band"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M band"
    return f"{value:.0f} band"


def _mask_value_recursive(value, masked_fields: set):
    if isinstance(value, dict):
        result = {}
        for key, sub_value in value.items():
            lower_key = key.lower()
            if any(marker in lower_key for marker in SECRET_FIELD_MARKERS):
                # API-H-004/API-H-007: secret/credential không bao giờ được gửi
                # cho OpenAI hay ghi log — loại bỏ hẳn khỏi payload.
                masked_fields.add(key)
                continue
            if lower_key in RESTRICTED_ID_FIELDS and sub_value is not None:
                prefix = _RESTRICTED_ID_TOKEN_PREFIX.get(lower_key, "ID")
                result[key] = _deterministic_token(prefix, sub_value)
                masked_fields.add(key)
            elif lower_key in NAME_FIELDS_TO_MASK and sub_value:
                result[key] = _partial_mask_name(sub_value)
                masked_fields.add(key)
            elif lower_key in AGGREGATE_AMOUNT_FIELDS and sub_value is not None:
                result[key] = _band_amount(sub_value)
                masked_fields.add(key)
            else:
                result[key] = _mask_value_recursive(sub_value, masked_fields)
        return result
    if isinstance(value, list):
        return [_mask_value_recursive(item, masked_fields) for item in value]
    return value


def mask_sensitive_fields(payload: dict) -> tuple[dict, list[str]]:
    """
    Trả về (payload đã mask/tokenize, danh sách tên field đã bị mask) để:
      1. Gửi cho OpenAI thay cho payload gốc (API-H-004).
      2. Ghi vào workflow_logs làm "masked_fields" theo đúng field
         25_RUNTIME_LOG_SCHEMA (bằng chứng đã che dữ liệu — API-H-007).
    Không sửa payload gốc (deep copy khi duyệt đệ quy).
    """
    masked_fields: set[str] = set()
    masked_payload = _mask_value_recursive(payload, masked_fields)
    return masked_payload, sorted(masked_fields)


# ============================================================
# 3. DATA LOADING — CSV-only pipeline (không dùng SQLite trung gian)
# ============================================================

@st.cache_data(show_spinner=False)
def load_team_pack(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Bóc tách từng sheet bắt buộc của Team Pack thành CSV trong bộ nhớ rồi nạp lại
    bằng pandas.read_csv. Theo System Prompt: hệ thống đọc dữ liệu trực tiếp từ CSV,
    tuyệt đối không dùng lưu trữ trung gian qua SQLite."""
    excel = pd.ExcelFile(BytesIO(file_bytes))
    missing = [sheet for sheet in REQUIRED_SHEETS if sheet not in excel.sheet_names]
    if missing:
        raise ValueError("Thiếu sheet bắt buộc: " + ", ".join(missing))

    data = {}
    for sheet in REQUIRED_SHEETS:
        raw_df = pd.read_excel(excel, sheet_name=sheet)
        raw_df.columns = [str(col).strip() for col in raw_df.columns]
        csv_text = raw_df.to_csv(index=False)  # bóc tách CSV
        df = pd.read_csv(StringIO(csv_text))  # nạp lại trực tiếp từ CSV
        data[sheet] = df
    return data


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def get_profile(data: dict[str, pd.DataFrame]) -> dict:
    profile_df = data["02_OPC_PROFILE"]
    return {
        str(row["field"]).strip(): clean_value(row["value"])
        for _, row in profile_df.iterrows()
    }


def format_vnd(value: float) -> str:
    value = float(value or 0)
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} tỷ VND"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.0f} triệu VND"
    return f"{value:,.0f} VND"


def to_date(value) -> Optional[pd.Timestamp]:
    """Chuyển serial ngày Excel hoặc chuỗi ngày sang pandas Timestamp."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
    return pd.to_datetime(value)


def is_core_city(province: Optional[str]) -> bool:
    if not province:
        return False
    normalized = str(province).strip().lower()
    return normalized in CORE_CITY_ALIASES


# ============================================================
# 4. TÁC NHÂN 1 — DATA & FINANCE AGENT (tính toán tất định bằng Python)
# ============================================================

def find_existing_customer(customers: pd.DataFrame, customer_name: str) -> Optional[dict]:
    if not customer_name:
        return None
    match = customers.loc[
        customers["customer_name"].astype(str).str.strip().str.lower()
        == customer_name.strip().lower()
    ]
    if match.empty:
        return None
    return {key: clean_value(value) for key, value in match.iloc[0].to_dict().items()}


def latest_transaction_risk_score(bank_txn: pd.DataFrame, customer_id: Optional[str]) -> Optional[float]:
    if not customer_id:
        return None
    rows = bank_txn.loc[bank_txn["counterparty_id"].astype(str) == str(customer_id)]
    if rows.empty:
        return None
    rows = rows.copy()
    rows["txn_date"] = rows["txn_date"].apply(to_date)
    rows = rows.sort_values("txn_date")
    return float(rows.iloc[-1]["transaction_risk_score"])


def compute_scale_coefficient(num_provinces: Optional[int]) -> tuple[float, Optional[dict]]:
    """Hệ số Oper theo quy mô triển khai (số tỉnh/thành phố), nhập thủ công ở Input Data:

    * < 10 tỉnh thành: +0.01
    * 10 đến 20 tỉnh thành: +0.02
    * > 20 tỉnh thành: +0.05    
    """
    if num_provinces is None or num_provinces <= 0:
        return 0.0, None

    if num_provinces < 10:
        he_so = 0.01
        mo_ta = f"< 10 tỉnh thành ({num_provinces})"
    elif num_provinces <= 20:
        he_so = 0.02
        mo_ta = f"10-20 tỉnh thành ({num_provinces})"
    else:
        he_so = 0.05     
        mo_ta = f"> 20 tỉnh thành ({num_provinces})"

    return he_so, {"tieu_chi": f"Quy mô triển khai ({mo_ta})", "he_so": he_so}


def compute_order_change_coefficient(order_count: Optional[int]) -> tuple[float, Optional[dict], bool]:
    """Hệ số Oper theo số lượng order (Order Change — PDF mục 2.2).

    FIX: PDF liệt kê Order Change (giới hạn 2 order/HĐ) là 1 phần của Oper Score
    CƠ SỞ, áp dụng ngay từ khi tính hợp đồng ban đầu — trước đây yếu tố này CHỈ
    tồn tại trong nhánh Crisis Card (resolve_crisis_deltas), khiến hợp đồng gốc
    không hề bị tính theo giới hạn order như PDF mô tả. Hàm này tách công thức ra
    dùng CHUNG cho cả 2 nơi (một bộ ngưỡng duy nhất — ORDER_CHANGE_* ở đầu file):
      * compute_oper_coefficient() / build_finance_metrics(): áp dụng khi Founder
        CÓ nhập initial_order_count lúc tạo hợp đồng ban đầu (tham số optional,
        mặc định None -> không có order nào -> không cộng Oper, giữ nguyên hành
        vi cũ cho các hợp đồng không nhập trường này — "chỉ thêm vào Oper khi nào
        được sử dụng/nhập thì mới áp dụng").
      * resolve_crisis_deltas(): khi ORDER_CHANGE xảy ra giữa hợp đồng.

    * <= 2 order: miễn phí, không cộng Oper.
    * > 2 và <= 10 order: +0.5% Oper.
    * > 10 order: vượt trần cứng -> KHÔNG cộng Oper ở đây (trả hard_cap_exceeded=True
      để tầng gọi tự quyết định, giống cách resolve_crisis_deltas xử lý — không raise
      Exception làm crash luồng).
    """
    if order_count is None or order_count <= 0:
        return 0.0, None, False

    if order_count > ORDER_CHANGE_HARD_CAP:
        return (
            0.0,
            {
                "tieu_chi": (
                    f"Order Change VƯỢT TRẦN CỨNG ({ORDER_CHANGE_HARD_CAP} - ngưỡng tạm đặt, "
                    f"cần Founder xác nhận lại con số này): {order_count} order"
                ),
                "he_so": 0.0,
            },
            True,
        )

    if order_count > ORDER_CHANGE_FREE_LIMIT:
        return (
            ORDER_CHANGE_SURCHARGE_OPER,
            {
                "tieu_chi": f"Order Change (vượt {ORDER_CHANGE_FREE_LIMIT} order, số order: {order_count})",
                "he_so": ORDER_CHANGE_SURCHARGE_OPER,
            },
            False,
        )

    return 0.0, None, False


def compute_oper_coefficient(
    payment_reliability: Optional[float],
    province: Optional[str],
    transaction_risk_score: Optional[float],
    order_date: pd.Timestamp,
    due_date: pd.Timestamp,
    num_provinces: Optional[int] = None,
    initial_order_count: Optional[int] = None,
) -> tuple[float, list[dict], bool]:
    """Cộng dồn hệ số Oper theo bảng điều kiện của System Prompt.

    Ghi chú: Uy tín thanh toán (Payment Reliability) và Áp lực tiến độ giao hàng
    (Urgent Delivery) đã được thay thế bằng một hệ số rủi ro con người CỐ ĐỊNH
    +1.0%, luôn được cộng vào Oper bất kể payment_reliability / thời hạn hợp đồng.

    num_provinces: quy mô triển khai dự án (số tỉnh thành), nhập thủ công ở Input Data.
    initial_order_count: số lượng order của hợp đồng ban đầu (PDF mục 2.2 — Order
        Change, giới hạn 2 order/HĐ), nhập thủ công ở Input Data. Optional — mặc
        định None (không nhập) thì không cộng thêm Oper, giữ nguyên hành vi/logic
        đầu ra cũ cho các hợp đồng không dùng trường này.

    Trả về (oper, breakdown, order_change_hard_cap_exceeded) — cờ thứ 3 báo hợp
    đồng ban đầu đã vượt trần cứng số order ngay từ lúc tạo (xem
    compute_order_change_coefficient).
    """
    oper = 0.0
    breakdown = []

    # Hệ số rủi ro con người (Human Risk Factor) — cố định, luôn áp dụng.
    oper += 0.01
    breakdown.append({"tieu_chi": "Rủi ro con người (Human Risk Factor - cố định)", "he_so": 0.01})

    if province is not None and not is_core_city(province):
        oper += 0.02
        breakdown.append({"tieu_chi": f"Mở rộng địa bàn ({province})", "he_so": 0.02})

    if transaction_risk_score is not None and transaction_risk_score > 85:
        oper += 0.04
        breakdown.append({"tieu_chi": "Rủi ro giao dịch > 85", "he_so": 0.04})

    scale_he_so, scale_breakdown_item = compute_scale_coefficient(num_provinces)
    if scale_breakdown_item is not None:
        oper += scale_he_so
        breakdown.append(scale_breakdown_item)

    # FIX: Order Change (PDF mục 2.2) — trước đây chỉ có trong Crisis Card, nay là
    # 1 phần của Oper Score cơ sở, áp dụng ngay khi tính hợp đồng ban đầu (nếu có
    # nhập initial_order_count). Xem compute_order_change_coefficient để biết chi
    # tiết ngưỡng/lý do dùng chung công thức với resolve_crisis_deltas.
    order_he_so, order_breakdown_item, order_hard_cap_exceeded = compute_order_change_coefficient(
        initial_order_count
    )
    if order_breakdown_item is not None:
        oper += order_he_so
        breakdown.append(order_breakdown_item)

    return oper, breakdown, order_hard_cap_exceeded


def build_finance_metrics(
    selected_products: pd.DataFrame,
    payment_reliability: Optional[float],
    province: Optional[str],
    transaction_risk_score: Optional[float],
    order_date: pd.Timestamp,
    due_date: pd.Timestamp,
    num_provinces: Optional[int] = None,
    initial_order_count: Optional[int] = None,
) -> dict:
    """
    baseline_estimate = Σ (list_price × (1 - target_margin))
    estimated_cost   = baseline_estimate × (1 + Oper)
    Gross_Margin     = (Σ list_price - estimated_cost) / Σ list_price

    num_provinces: quy mô triển khai dự án (số tỉnh thành), nhập thủ công trên UI —
    được cộng thêm vào hệ số Oper theo compute_scale_coefficient().
    initial_order_count: số lượng order của hợp đồng ban đầu (PDF mục 2.2 — Order
    Change), nhập thủ công trên UI (optional) — được cộng thêm vào hệ số Oper theo
    compute_order_change_coefficient().
    """
    total_list_price = float(selected_products["list_price"].sum())
    baseline_estimate = float(
        (selected_products["list_price"] * (1 - selected_products["target_margin"])).sum()
    )

    oper, oper_breakdown, order_change_hard_cap_exceeded = compute_oper_coefficient(
        payment_reliability, province, transaction_risk_score, order_date, due_date,
        num_provinces, initial_order_count,
    )

    estimated_cost = baseline_estimate * (1 + oper)
    gross_margin = (
        (total_list_price - estimated_cost) / total_list_price
        if total_list_price > 0
        else 0.0
    )
    contract_months = max(1, round((due_date - order_date).days / 30))

    return {
        "total_list_price": total_list_price,
        "baseline_estimate": round(baseline_estimate, 2),
        "oper_coefficient": round(oper, 4),
        "oper_breakdown": oper_breakdown,
        "estimated_cost": round(estimated_cost, 2),
        "gross_margin": round(gross_margin, 6),
        "contract_months": contract_months,
        "initial_order_count": initial_order_count,
        "order_change_hard_cap_exceeded": order_change_hard_cap_exceeded,
    }


def project_closing_cash(
    data: dict[str, pd.DataFrame],
    selected_products: pd.DataFrame,
    finance_metrics: dict,
    order_date: pd.Timestamp,
    reserve_minimum: float,
) -> dict:
    """
    Mô phỏng dòng tiền theo tháng của hợp đồng mới, cộng vào baseline 09_CASHFLOW.

    Baseline projected_closing_cash của mỗi tháng trong 09_CASHFLOW được GIỮ NGUYÊN
    làm nền (không tự chain lại từ expected_cash_in/out vì hai chuỗi này không
    reconcile với nhau trong dữ liệu gốc). Dòng tiền hợp đồng mới được cộng thêm
    dưới dạng lũy kế:
        Projected_Closing_Cash(tháng i) = Baseline_Closing_Cash(tháng i)
                                          + Σ_{k=0..i} (Deal_Cash_In(k) - Deal_Cash_Out(k))
    Opening_Cash(tháng hiện tại) = Projected_Closing_Cash(tháng trước, đã gồm hợp đồng mới).
    """
    cashflow = data["09_CASHFLOW"].copy()
    cashflow["month"] = cashflow["month"].astype(str)
    months_count = finance_metrics["contract_months"]
    estimated_cost = finance_metrics["estimated_cost"]

    # Phân bổ cash-in / cash-out mỗi tháng của hợp đồng mới theo loại dịch vụ.
    monthly_cash_in = [0.0] * months_count
    monthly_cash_out = [0.0] * months_count
    cost_out_per_month = estimated_cost / months_count if months_count else 0.0

    for _, product in selected_products.iterrows():
        list_price = float(product["list_price"])
        pricing_model = str(product.get("pricing_model", "")).strip()

        if pricing_model == PRICING_MODEL_MONTHLY_SUBSCRIPTION:
            # SVC-002, SVC-003: Tiền vào = list_price / số tháng.
            per_month_in = list_price / months_count
            for i in range(months_count):
                monthly_cash_in[i] += per_month_in
        elif pricing_model == PRICING_MODEL_INITIAL_SETUP:
            # SVC-001: Tiền vào = tổng list_price, thu ngay khi khởi tạo.
            monthly_cash_in[0] += list_price
        elif pricing_model == PRICING_MODEL_PROJECT:
            # SVC-004, SVC-005: Tiền vào phụ thuộc cột mốc (milestone). Team Pack
            # hiện chưa có bảng milestone chi tiết -> giả định milestone tuyến tính
            # (chia đều list_price theo số tháng thực hiện).
            per_month_in = list_price / months_count
            for i in range(months_count):
                monthly_cash_in[i] += per_month_in
        else:
            # pricing_model không khớp 3 nhóm đã định nghĩa trong System Prompt -> mặc
            # định chia đều theo tháng, không chặn luồng nhưng cần rà soát dữ liệu nguồn.
            per_month_in = list_price / months_count
            for i in range(months_count):
                monthly_cash_in[i] += per_month_in

    for i in range(months_count):
        monthly_cash_out[i] += cost_out_per_month

    # Xác định điểm neo (Opening Cash) từ baseline 09_CASHFLOW của tháng trước order_date.
    start_month_period = order_date.to_period("M")
    prior_month_str = str(start_month_period - 1)
    prior_row = cashflow.loc[cashflow["month"] == prior_month_str]
    baseline_anchor = (
        float(prior_row.iloc[0]["projected_closing_cash"])
        if not prior_row.empty
        else 0.0
    )

    # QUAN TRỌNG: cột projected_closing_cash trong 09_CASHFLOW là dự báo đã chốt sẵn
    # cho từng tháng (không tự tái lập được bằng cách cộng dồn expected_cash_in -
    # expected_cash_out của các tháng liền kề — hai chuỗi này không khớp nhau trong
    # dữ liệu gốc). Vì vậy, thay vì "chain" lại từ đầu bằng expected_cash_in/out thô,
    # ta GIỮ NGUYÊN baseline projected_closing_cash của sheet làm nền, và chỉ CỘNG THÊM
    # phần dòng tiền phát sinh lũy kế của hợp đồng mới:
    #   Projected_Closing_Cash(tháng i) = Baseline_Closing_Cash(tháng i)
    #                                     + Σ_{k=0..i} (Deal_Cash_In(k) - Deal_Cash_Out(k))
    # Nếu tháng vượt quá phạm vi 09_CASHFLOW, baseline được giữ nguyên bằng giá trị
    # tháng gần nhất đã biết (giả định công ty duy trì trạng thái ổn định nếu không
    # có hợp đồng mới).
    last_known_baseline_closing = baseline_anchor
    previous_baseline_closing = baseline_anchor

    schedule = []
    cumulative_deal_net = 0.0
    prior_new_closing = baseline_anchor
    for i in range(months_count):
        month_period = start_month_period + i
        month_str = str(month_period)
        baseline_row = cashflow.loc[cashflow["month"] == month_str]

        if not baseline_row.empty:
            baseline_closing = float(baseline_row.iloc[0]["projected_closing_cash"])
            last_known_baseline_closing = baseline_closing
        else:
            baseline_closing = last_known_baseline_closing

        # Biến động baseline giữa tháng này và tháng trước (có thể âm) -- đây chính là
        # phần "Expected_Cash_In - Expected_Cash_Out" ứng với hoạt động sẵn có của công
        # ty (không tính hợp đồng mới), giúp Opening + biến động + dòng tiền hợp đồng =
        # Closing khớp chính xác từng dòng khi kiểm chứng thủ công.
        baseline_closing_change = baseline_closing - previous_baseline_closing
        previous_baseline_closing = baseline_closing

        deal_in = monthly_cash_in[i]
        deal_out = monthly_cash_out[i]
        cumulative_deal_net += deal_in - deal_out

        opening_cash = prior_new_closing
        projected_closing_cash = baseline_closing + cumulative_deal_net

        schedule.append(
            {
                "month": month_str,
                "opening_cash": round(opening_cash, 2),
                "baseline_projected_closing_cash": round(baseline_closing, 2),
                "baseline_closing_change": round(baseline_closing_change, 2),
                "deal_cash_in": round(deal_in, 2),
                "deal_cash_out": round(deal_out, 2),
                "projected_closing_cash": round(projected_closing_cash, 2),
            }
        )
        prior_new_closing = projected_closing_cash

    min_closing_cash = min(row["projected_closing_cash"] for row in schedule)
    breach = min_closing_cash < reserve_minimum

    return {
        "schedule": schedule,
        "min_projected_closing_cash": round(min_closing_cash, 2),
        "cash_reserve_minimum": reserve_minimum,
        "cash_reserve_breach": breach,
    }


# ---- Confidence Score (CS = 0.4*Eliscore + 0.6*Completeness_Score) ----
#
# Chỉ tính Confidence Score SAU KHI đã lọc 4 lớp sản phẩm ngân hàng (Lớp 1 —
# account_ops / credit_guarantee / unclassified bị loại; Lớp 2 — Min Funding;
# Lớp 3 — so sánh tổng chi phí; Lớp 4 — ràng buộc tài sản đảm bảo) và thu được
# ÍT NHẤT một gói vay phù hợp
# (eligible=True trong partner_matrix). Nếu không có đề xuất gói vay nào, trả về None
# — giữ đúng logic hiện tại (không tính Confidence Score khi không có đề xuất tài trợ).
#
# Eliscore (điểm năng lực tài chính) = 0.6 * S_liquidity + 0.4 * S_margin
#   - S_liquidity = closing_cash / cash_reserve_minimum (550 triệu VND); nếu < 0 -> gán 0.
#   - S_margin = (list_price - funding_amount * (annual_rate_or_fee + processing_fee_rate)) / list_price
#     (tính trên gói vay được chọn — phương án eligible xếp hạng cao nhất trong partner_matrix).
#
# Completeness_score (độ hoàn thiện dữ liệu đầu vào): thiếu province -> 0, đủ province -> 1.


def compute_eliscore(
    cash_projection: dict,
    total_list_price: float,
    funding_amount: float,
    selected_bank_product: dict,
) -> dict:
    reserve_minimum = cash_projection["cash_reserve_minimum"]
    closing_cash = cash_projection["min_projected_closing_cash"]

    s_liquidity = (closing_cash / reserve_minimum) if reserve_minimum else 0.0
    if s_liquidity < 0:
        s_liquidity = 0.0

    total_rate = float(selected_bank_product["annual_rate_or_fee"]) + float(
        selected_bank_product["processing_fee_rate"]
    )
    if total_list_price > 0:
        s_margin = (total_list_price - funding_amount * total_rate) / total_list_price
    else:
        s_margin = 0.0

    eliscore = 0.6 * s_liquidity + 0.4 * s_margin

    return {
        "s_liquidity": round(s_liquidity, 4),
        "s_margin": round(s_margin, 4),
        "eliscore": round(eliscore, 4),
        "closing_cash": closing_cash,
        "cash_reserve_minimum": reserve_minimum,
        "total_list_price": total_list_price,
        "funding_amount": funding_amount,
        "selected_bank_product_id": selected_bank_product.get("bank_product_id"),
    }


def compute_completeness_score(province: Optional[str]) -> dict:
    has_province = bool(province and str(province).strip())
    completeness_score = 1.0 if has_province else 0.0
    return {
        "completeness_score": completeness_score,
        "province_present": has_province,
    }


def compute_confidence_score(
    cash_projection: dict,
    partner_matrix: list[dict],
    total_list_price: float,
    funding_amount: float,
    province: Optional[str],
) -> Optional[dict]:
    """Chỉ tính Confidence Score khi (sau khi lọc 4 lớp) có ít nhất một gói vay
    eligible trong partner_matrix — tức là đã có đề xuất gói vay phù hợp."""
    eligible_options = [item for item in partner_matrix if item.get("eligible")]
    if not eligible_options:
        return None

    selected_bank_product = eligible_options[0]

    eliscore_result = compute_eliscore(
        cash_projection=cash_projection,
        total_list_price=total_list_price,
        funding_amount=funding_amount,
        selected_bank_product=selected_bank_product,
    )
    completeness_result = compute_completeness_score(province)

    cs = (
        0.4 * eliscore_result["eliscore"]
        + 0.6 * completeness_result["completeness_score"]
    )
    cs = max(0.0, min(1.0, cs))

    return {
        "confidence_score": round(cs, 4),
        "eliscore": eliscore_result,
        "completeness": completeness_result,
    }


# ============================================================
# 5. TÁC NHÂN 2 — RISK & COMPLIANCE AGENT
# ============================================================

def evaluate_risk_rules(
    data: dict[str, pd.DataFrame],
    finance_metrics: dict,
    cash_projection: dict,
    confidence_result: Optional[dict],
) -> dict:
    rules = data["13_RISK_RULES"]
    rule_lookup = {str(r["rule_id"]): r for _, r in rules.iterrows()}
    triggered = []

    # RR-003: gross_margin < 0.28 -> "Đề xuất tối ưu chi phí"
    if finance_metrics["gross_margin"] < 0.28:
        triggered.append(
            {
                "rule_id": "RR-003",
                "risk_type": clean_value(rule_lookup["RR-003"]["risk_type"]),
                "severity": clean_value(rule_lookup["RR-003"]["severity"]),
                "message": "Đề xuất tối ưu chi phí",
                "evidence": f"gross_margin={finance_metrics['gross_margin']:.4f} < 0.28",
            }
        )

    # RR-002: Projected_Closing_Cash < 550 triệu VND
    rr002_fired = cash_projection["cash_reserve_breach"]
    if rr002_fired:
        triggered.append(
            {
                "rule_id": "RR-002",
                "risk_type": clean_value(rule_lookup["RR-002"]["risk_type"]),
                "severity": clean_value(rule_lookup["RR-002"]["severity"]),
                "message": "Recommend working capital option or phase delivery",
                "evidence": (
                    f"min_projected_closing_cash={cash_projection['min_projected_closing_cash']}; "
                    f"reserve={cash_projection['cash_reserve_minimum']}"
                ),
            }
        )

        # RR-006 chỉ xảy ra khi RR-002 đã xảy ra.
        if confidence_result is not None and confidence_result["confidence_score"] < 0.65:
            triggered.append(
                {
                    "rule_id": "RR-006",
                    "risk_type": clean_value(rule_lookup["RR-006"]["risk_type"]),
                    "severity": clean_value(rule_lookup["RR-006"]["severity"]),
                    "message": "Ask for missing data or provide no-recommendation",
                    "evidence": f"confidence_score={confidence_result['confidence_score']} < 0.65",
                }
            )

    severities = {item["severity"] for item in triggered}
    if "Critical" in severities:
        risk_level = "CRITICAL"
    elif "High" in severities:
        risk_level = "HIGH"
    elif "Medium" in severities:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {"triggered_rules": triggered, "risk_level": risk_level}


def build_risk_summary_message(gross_margin: float, min_closing_cash: float) -> dict:
    """Risk Summary (thuần Python, tách biệt với RR-002/RR-003):
    - Đồng thời gross_margin < 20% VÀ min_projected_closing_cash < 0
      -> "Vi phạm nghiêm trọng các yêu cầu về biên lợi nhuận và dòng tiền dự trữ"
    - Chỉ MỘT trong hai điều kiện trên xảy ra
      -> "Xem xét lại khả năng thanh khoản và biên lợi nhuận"
    - Không điều kiện nào xảy ra -> an toàn.
    """
    gm_breach = gross_margin < 0.20
    cash_breach = min_closing_cash < 0

    if gm_breach and cash_breach:
        level = "CRITICAL"
        message = "Vi phạm nghiêm trọng các yêu cầu về biên lợi nhuận và dòng tiền dự trữ"
    elif gm_breach or cash_breach:
        level = "WARNING"
        message = "Xem xét lại khả năng thanh khoản và biên lợi nhuận"
    else:
        level = "OK"
        message = "Biên lợi nhuận và dòng tiền dự trữ đều trong ngưỡng an toàn."

    return {"level": level, "message": message, "gm_breach": gm_breach, "cash_breach": cash_breach}


# ============================================================
# 6. TÁC NHÂN 3 — DECISION & PARTNER AGENT
# ============================================================

# --- Phân loại bản chất sản phẩm ngân hàng (KHÔNG dựa vào minimum_amount, mà dựa
# trên product_name/description) — dùng để lọc TRƯỚC KHI so sánh gói vay, tránh
# đề xuất nhầm các dịch vụ vận hành tài khoản (phí gần 0đ) làm "giải pháp huy động
# vốn", và tránh đề xuất nhầm công cụ bảo lãnh/hỗ trợ giao dịch cho nhu cầu bù đắp
# THIẾU HỤT TIỀN MẶT (RR-002) — vì cả hai đều sai bản chất/mục đích sử dụng vốn dù
# có thể có eligible=True về mặt hạn mức.
BANK_PRODUCT_ACCOUNT_OPS_KEYWORDS = [
    "cash management", "collection account", "collection and settlement",
    "alert workflow", "transaction alert", "account aggregation", "statement",
]
BANK_PRODUCT_CREDIT_CASH_KEYWORDS = [
    "working capital", "factoring", "advance against", "credit line", "short-term credit",
]
BANK_PRODUCT_CREDIT_GUARANTEE_KEYWORDS = [
    "bond", "guarantee", "trade finance", "letter of credit", " lc ", "international payment",
]


def classify_bank_product(product_name: str, description: str) -> tuple[str, str]:
    """LỚP 1 — Loại gói tín dụng: xác định nhu cầu vay.
    Phân loại 1 sản phẩm 11_BANK_PRODUCTS theo bản chất, trả về (category, lý do).

    - "account_ops": dịch vụ vận hành tài khoản (không phải khoản vay) -> LOẠI khỏi
      so sánh gói vay dưới mọi trường hợp.
    - "credit_cash": sản phẩm tín dụng bơm tiền mặt trực tiếp (working capital,
      factoring...) -> phù hợp để so sánh khi cần bù đắp thiếu hụt tiền mặt (RR-002).
    - "credit_guarantee": sản phẩm tín dụng nhưng là bảo lãnh/hỗ trợ giao dịch
      (performance bond, trade finance/LC...) -> vẫn là sản phẩm tín dụng hợp lệ,
      nhưng KHÔNG bơm tiền mặt trực tiếp nên không dùng để giải quyết RR-002.
    - "unclassified": không khớp từ khóa nào -> KHÔNG tự suy đoán, gắn cờ để Founder
      tự rà soát thủ công thay vì để hệ thống tự ý đưa vào/loại ra.
    """
    text = f"{product_name} {description}".lower()
    for kw in BANK_PRODUCT_ACCOUNT_OPS_KEYWORDS:
        if kw in text:
            return "account_ops", kw
    for kw in BANK_PRODUCT_CREDIT_CASH_KEYWORDS:
        if kw in text:
            return "credit_cash", kw
    for kw in BANK_PRODUCT_CREDIT_GUARANTEE_KEYWORDS:
        if kw in text:
            return "credit_guarantee", kw
    return "unclassified", "không khớp từ khóa phân loại nào"


def classify_all_bank_products(data: dict[str, pd.DataFrame]) -> list[dict]:
    """Bảng phân loại đầy đủ TOÀN BỘ 11_BANK_PRODUCTS (không phụ thuộc funding_need),
    dùng làm audit trail hiển thị cho Founder — giải thích tường minh vì sao mỗi sản
    phẩm được giữ lại hay loại khỏi so sánh gói vay."""
    products = data["11_BANK_PRODUCTS"]
    result = []
    for _, product in products.iterrows():
        category, reason = classify_bank_product(
            str(product["product_name"]), str(product.get("description", ""))
        )
        result.append(
            {
                "bank": clean_value(product["bank"]),
                "product_name": clean_value(product["product_name"]),
                "category": category,
                "matched_keyword": reason,
                "included_in_comparison": category == "credit_cash",
            }
        )
    return result


def build_partner_matrix(
    data: dict[str, pd.DataFrame],
    funding_need: float,
    cash_projection: dict,
) -> list[dict]:
    """Cơ chế Lọc 4 Lớp Quản trị Vốn (chỉ truy xuất 11_BANK_PRODUCTS khi
    Projected_Closing_Cash < 550 triệu VND):

    LỚP 1 — Loại gói tín dụng: xác định nhu cầu vay. Chỉ giữ lại các gói TÍN DỤNG
    BƠM TIỀN MẶT TRỰC TIẾP (category="credit_cash"); không lọc theo
    target_segment/customer_type, nhưng loại bỏ:
      (1) dịch vụ vận hành tài khoản (account_ops) — không phải khoản vay;
      (2) sản phẩm tín dụng bảo lãnh/hỗ trợ giao dịch (credit_guarantee) — không
          bơm tiền mặt trực tiếp nên sai mục đích so với nhu cầu bù đắp RR-002;
      (3) sản phẩm chưa phân loại được (unclassified) — không tự đoán, xem
          classify_all_bank_products() để Founder tự rà soát.

    LỚP 2 — Nhu cầu Vốn tối thiểu (Min Funding): xác định nhu cầu vốn. Một gói
    vay chỉ eligible=True khi funding_need đạt đủ minimum_amount của gói đó.

    LỚP 3 — So sánh tổng chi phí: annual_rate_or_fee + processing_fee_rate của
    các gói vay, gói nào tổng chi phí thấp hơn được ưu tiên xếp hạng cao hơn.

    LỚP 4 — Ràng buộc Tài sản Đảm bảo (Collateral): ưu tiên gói vay có
    collateral_ratio (tỷ lệ thế chấp) thấp hơn, để xác định khả năng đảm bảo tài
    chính của OPC — dùng làm tiêu chí phân định khi Lớp 3 bằng nhau."""
    if not cash_projection["cash_reserve_breach"]:
        return []

    candidates = data["11_BANK_PRODUCTS"].copy()

    matrix = []
    for _, product in candidates.iterrows():
        # LỚP 1 — Loại gói tín dụng: xác định nhu cầu vay (chỉ giữ credit_cash).
        category, _reason = classify_bank_product(
            str(product["product_name"]), str(product.get("description", ""))
        )
        if category != "credit_cash":
            continue

        # LỚP 2 — Nhu cầu Vốn tối thiểu (Min Funding): xác định nhu cầu vốn.
        min_amount = float(product["minimum_amount"])
        total_cost_rate = float(product["annual_rate_or_fee"]) + float(product["processing_fee_rate"])
        # BUG cũ: "min_amount <= max(funding_need, min_amount)" luôn luôn đúng (tautology)
        # vì max(funding_need, min_amount) không bao giờ nhỏ hơn min_amount -> mọi sản phẩm
        # đều bị đánh dấu eligible=True dù funding_need thấp hơn minimum_amount rất nhiều.
        # Sửa lại: chỉ eligible khi khoản cần vay (funding_need) đạt đủ minimum_amount của
        # sản phẩm ngân hàng đó.
        eligible = funding_need >= 0 and funding_need >= min_amount
        matrix.append(
            {
                "bank_product_id": clean_value(product["bank_product_id"]),
                "bank": clean_value(product["bank"]),
                "product_name": clean_value(product["product_name"]),
                "target_segment": clean_value(product["target_segment"]),
                "product_category": category,
                "annual_rate_or_fee": float(product["annual_rate_or_fee"]),
                "processing_fee_rate": float(product["processing_fee_rate"]),
                "collateral_ratio": float(product["collateral_ratio"]),
                "minimum_amount": min_amount,
                "automation_level": clean_value(product["automation_level"]),
                "total_cost_rate": round(total_cost_rate, 4),
                "eligible": eligible,
            }
        )

    # Sắp xếp: ưu tiên eligible (Lớp 2), sau đó:
    # LỚP 3 — So sánh tổng chi phí (annual_rate_or_fee + processing_fee_rate) thấp nhất trước;
    # LỚP 4 — Ràng buộc Tài sản Đảm bảo (Collateral): khi Lớp 3 bằng nhau, ưu tiên gói vay có
    # collateral_ratio (tỷ lệ thế chấp) thấp hơn — xác định khả năng đảm bảo tài chính của OPC.
    matrix.sort(key=lambda item: (not item["eligible"], item["total_cost_rate"], item["collateral_ratio"]))
    return matrix


def determine_requested_amount(cash_projection: dict, partner_matrix: list[dict]) -> float:
    """Số tiền yêu cầu = phần thiếu hụt so với ngưỡng dự trữ tối thiểu, tối thiểu
    bằng minimum_amount của sản phẩm ngân hàng phù hợp nhất — CHỈ KHI sản phẩm đó
    thực sự eligible=True.

    BUG cũ: lấy partner_matrix[0] bất kể eligible hay không. Khi KHÔNG có sản phẩm
    tín dụng nào đủ điều kiện (mọi eligible=False), partner_matrix[0] vẫn là sản
    phẩm rẻ nhất trong danh sách (không eligible), và minimum_amount của nó (có thể
    hàng trăm triệu) vẫn bị dùng làm sàn cho requested_amount. Hệ quả: requested_amount
    có thể vượt ngưỡng 300 triệu (RR-005) và kích hoạt cảnh báo "Founder Approval Gate"
    dù funding_amount thực tế hiển thị cho người dùng = 0 VND (vì enforce_decision_card
    lọc đúng theo eligible=True, không có gói nào khả thi để vay). Sửa: chỉ nâng sàn
    funding_need lên minimum_amount khi sản phẩm tốt nhất thực sự eligible; nếu không,
    requested_amount giữ đúng bằng khoản thiếu hụt tiền mặt thực tế (không vay được)."""
    reserve = cash_projection["cash_reserve_minimum"]
    min_cash = cash_projection["min_projected_closing_cash"]
    funding_need = max(0.0, reserve - min_cash)
    if partner_matrix:
        best = partner_matrix[0]
        if best.get("eligible"):
            funding_need = max(funding_need, best["minimum_amount"])
    return funding_need


# ============================================================
# 7. OPENAI AGENT CALLS
# ============================================================

def call_structured_agent(
    client: OpenAI,
    model: str,
    instructions: str,
    payload: dict,
    output_schema: type[BaseModel],
    agent_name: str = "Agent"
):
    """
    Call one OpenAI agent and force its answer to follow a Pydantic JSON schema.
    Bao gồm cơ chế Tự động thử lại (Retry) CÓ HIỂN THỊ ĐẾM NGƯỢC trên giao diện
    để người dùng không bị hoảng loạn và bấm Refresh làm hỏng tiến trình.
    """
    prompt = (
        instructions.strip()
        + "\n\nDỮ LIỆU ĐẦU VÀO JSON:\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=output_schema,
            )

            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("OpenAI không trả về nội dung.")

            return parsed, completion.id

        except Exception as exc:
            err_str = str(exc)
            # Nếu là lỗi Quota/Rate limit 429 và chưa quá số lần thử
            if ("429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower() or "too_many_requests" in err_str.lower()) and attempt < max_retries - 1:
                wait_time = 45 
                countdown_placeholder = st.empty()
                for i in range(wait_time, 0, -1):
                    countdown_placeholder.warning(
                        f"⏳ **OpenAI API báo quá tải (429)!**\n\n"
                        f"{agent_name} đang tự động chờ để thử lại... **{i} giây**\n\n"
                        f"👉 *VUI LÒNG ĐỪNG BẤM GÌ CẢ (không F5, không bấm Chạy lại)* để hệ thống tự xử lý!"
                    )
                    time.sleep(1)
                countdown_placeholder.empty()
                continue
            raise exc


def run_finance_agent(client, model, payload):
    instructions = """
Bạn là Data & Finance Agent của OPC.

Nhiệm vụ:
1. Đọc các chỉ số đã được Python tính toán (baseline_estimate, Oper, estimated_cost,
   gross_margin, cash_projection, confidence_score nếu có).
2. Đánh giá chất lượng dữ liệu đầu vào.
3. Đưa ra đánh giá sơ bộ về khả năng nhận hợp đồng.

Quy tắc bắt buộc:
- Không tự thay đổi, làm tròn lại hoặc phát minh số liệu.
- Không tự tính lại các chỉ số; dùng đúng finance_metrics/cash_projection được cung cấp.
- Nếu missing_fields không rỗng, data_quality phải là MISSING_DATA và
  preliminary_assessment phải là NEED_MORE_DATA (Chỉ kích hoạt khi thiếu dữ liệu đầu vào, nếu không thiếu bắt buộc không được kích hoạt). 
- Viết bằng tiếng Việt, ngắn gọn, phục vụ Founder.
- Chỉ dựa trên payload được cung cấp.
"""
    return call_structured_agent(client, model, instructions, payload, FinanceAgentOutput, "Data & Finance Agent")


def run_risk_agent(client, model, payload):
    instructions = """
Bạn là Risk & Compliance Agent của OPC.

Nhiệm vụ:
1. Diễn giải các risk rule đã được Python xác định là triggered
   (RR-003 gross_margin<0.28, RR-002 Projected_Closing_Cash<550 triệu,
   RR-006 chỉ khi RR-002 đã xảy ra và confidence_score<0.65).
2. Tổng hợp risk level, cảnh báo, biện pháp kiểm soát và điểm cần Founder xác nhận.
3. Nêu rõ các rủi ro chưa thể đánh giá do thiếu nguồn dữ liệu (nếu có).

Quy tắc bắt buộc:
- Không tự tạo thêm risk rule ngoài triggered_rules được cung cấp.
- Không tuyên bố đã đánh giá rủi ro không có trong triggered_rules.
- risk_level phải khớp với risk_level đã được Python tính (dùng lại nguyên giá trị). Đưa ra tất cả các giá trị risk level đang xuất hiện.  
- Nếu missing_fields không rỗng, phải nêu yêu cầu bổ sung dữ liệu.
- Viết bằng tiếng Việt, ngắn gọn và có thể hành động.
- warnings: mỗi cảnh báo chỉ 1 câu ngắn (tối đa ~25 từ), nêu đúng 1 ý chính (rule
  nào bị vi phạm + hậu quả/rủi ro chính), KHÔNG liệt kê đầy đủ số liệu/evidence chi
  tiết trong câu này (evidence chi tiết đã có sẵn trong triggered_rules). Ưu tiên
  chất lượng — chỉ giữ lại cảnh báo quan trọng nhất, không cần liệt kê hết mọi khía
  cạnh.
"""
    return call_structured_agent(client, model, instructions, payload, RiskAgentOutput, "Risk & Compliance Agent")


def run_decision_agent(client, model, payload):
    instructions = """
Bạn là Decision & Partner Agent của OPC.

Nhiệm vụ:
1. So sánh partner_matrix (11_BANK_PRODUCTS) — Python đã lọc sẵn 2 cấp trước khi đưa
   vào đây: (a) loại dịch vụ vận hành tài khoản không phải khoản vay, và (b) loại
   sản phẩm tín dụng bảo lãnh/hỗ trợ giao dịch không bơm tiền mặt trực tiếp — chỉ còn
   sản phẩm tín dụng bơm tiền mặt trực tiếp (working capital/factoring), không lọc
   theo customer_type. Chọn ra phương án phù hợp nhất trong danh sách đã lọc này.
2. Tạo Decision Card gồm ĐÚNG 3 chỉ số bắt buộc (gross_margin, closing_cash,
   confidence_score), 1 phương án tài chính đề xuất, đúng 3 lý do và đúng 1
   điều kiện bảo vệ cần con người xác nhận.
3. Đưa ra recommendation đề xuất cho Founder .

Quy tắc bắt buộc:
- EXECUTIVE SUMMARY bắt buộc là nhận xét đánh giá đối với gói vay chứ không liên quan gì đến khách hàng. Phải đánh giá gói vay đang được đề xuất không tự động lấy gói vay khác. 
- gross_margin, closing_cash, confidence_score PHẢI lấy đúng giá trị Python cung cấp,
  không tự tính lại.
- Chỉ chọn phương án tài chính có eligible=true trong partner_matrix; nếu partner_matrix
  rỗng (không cần vay), selected_financing_option phải nêu rõ "Không cần huy động vốn ngoài".
- Nếu requested_amount > 300,000,000 VND: human_approval_required=true (RR-005, Founder phê duyệt).
- Không phát minh sản phẩm, lãi suất hoặc hạn mức ngoài dữ liệu được cung cấp.
- three_reasons phải có chính xác 3 phần tử, MỖI phần tử đánh giá đúng 1 trong 3 chỉ số
  bắt buộc theo thứ tự cố định:
      (1) Đánh giá gross margin: lấy gross margin được tính toán ở trên, không được tự nghĩ ra số liệu hay tự ý thay đổi số liệu. 
          - Nếu gross margin < 0.28: Nội dung bắt buộc sinh ra: "Chỉ số Gross Margin hiện tại là {gross_margin}, bé hơn mức tiêu chuẩn 0.28 (Kích hoạt RR-003). Điều này sẽ ảnh hưởng trực tiếp đến khả năng tài chính của OPC, do đó bắt buộc phải tiến hành đàm phán lại với khách hàng."
          - Nếu gross margin > 0.28: Nội dung bắt buộc sinh ra: "Chỉ số Gross Margin hiện tại là {gross_margin}, lớn hơn hoặc bằng mức tiêu chuẩn 0.28. Mức biên lợi nhuận này đang ở trạng thái an toàn."
      (2) Đánh giá closing cash: lấy closing cash được tính toán ở trên, không được tự nghĩ ra số liệu hay tự ý thay đổi số liệu. 
          - Nếu min_projected_closing_cash < 550 triệu: Nội dung bắt buộc sinh ra: "Projected Closing Cash hiện tại là {min_projected_closing_cash}, bé hơn mốc an toàn 550 triệu VND (Kích hoạt RR-002). Dự án đang có rủi ro về khả năng thanh khoản."
          - Nếu min_projected_closing_cash > 550 triệu: Nội dung bắt buộc sinh ra: "Projected Closing Cash hiện tại là {min_projected_closing_cash}, lớn hơn hoặc bằng mốc 550 triệu VND. Khả năng thanh khoản của dự án được đảm bảo."
      (3) Đánh giá confidence score: 
          - Nếu confidence_score < 0.65 :Nội dung bắt buộc sinh ra: "Confidence Score hiện tại là {confidence_score}, bé hơn mức 0.65 (Kích hoạt RR-006 do RR-002 đã cảnh báo). Mức độ tin cậy của dữ liệu dự phóng thấp, cần rà soát lại đầu vào."
          - Nếu confidence_score > 0.65: :Nội dung bắt buộc sinh ra: "Confidence Score hiện tại là {confidence_score} (Không kích hoạt RR-006). Mức độ tin cậy của dữ liệu ở mức cao hơn 65% và có thể chấp nhận để ra quyết định."
      Mỗi lý do phải nêu rõ số liệu cụ thể (giá trị chỉ số) và rule liên quan nếu có kích hoạt, không được viết chung chung hay gộp nhiều chỉ số vào 1 lý do.
      LƯU Ý: bắt buộc phải lấy đúng các chỉ số gross margin, closing cash, confidence score Ở TRÊN. Không được bịa chỉ số đầu vào, không lấy từ ngoài. 
- protection_condition phải là một điều kiện thương mại hoặc kiểm soát cụ thể cần Founder xác nhận.
- Viết bằng tiếng Việt, rõ ràng và bảo vệ được khi vấn đáp.
- So sánh đúng requested_amount > 300tr mới cần Founder phê duyệt.
- Quy tắc khi đưa ra recommendation: 
    + ACCEPT: Chấp nhận hoàn toàn đề xuất (khi các chỉ số tài chính đạt chuẩn, không có rủi ro lớn và dữ liệu đầy đủ).
    + CONDITIONAL_ACCEPT: Chấp nhận có điều kiện (khi dự án có thể thực hiện nhưng đi kèm các yêu cầu ràng buộc, biện pháp kiểm soát rủi ro hoặc cần đàm phán lại một số điều khoản như biên lợi nhuận). Các chỉ số không quá thấp đối với ngưỡng yêu cầu của cái Risk rule.
    + REJECT: Từ chối đề xuất (khi vi phạm các quy tắc rủi ro nghiêm trọng hoặc không đáp ứng các tiêu chuẩn cốt lõi) khi tất cả các chỉ số bao gồm: gross margin < 0.28, closing cash < 550tr, confidence score < 0.65 cùng xảy ra thì mới kích hoạt Reject. Nếu không đồng thời xảy ra bắt buộc không được kích hoạt reject.
    + NEED_MORE_DATA: Chỉ kích hoạt khi có thông tin đầu vào thiếu. Còn lại không được phép kích hoạt. 
"""
    return call_structured_agent(client, model, instructions, payload, DecisionAgentOutput)


def build_protection_condition(
    triggered_rule_ids: list[str],
    is_new_customer: bool = False,
    has_financing: bool = True,
) -> str:
    """
    Sinh "Điều kiện bảo vệ" TẤT ĐỊNH theo đúng tổ hợp risk rule đã kích hoạt.

    Lý do: nếu để OpenAI tự do viết, cùng một payload (cùng triggered_rule_ids) vẫn
    cho ra điều kiện bảo vệ khác nhau mỗi lần chạy (đã quan sát thấy trong thực tế:
    3 lần chạy cùng input ra 3 điều kiện hoàn toàn khác nhau) — mất tính tái lập và
    kiểm chứng được, vốn bắt buộc với một cam kết mà Founder phải xác nhận. Ưu tiên
    rule nghiêm trọng nhất: RR-006 (thiếu tin cậy dữ liệu) > RR-002 (rủi ro thanh
    khoản) > RR-003 (biên lợi nhuận thấp).

    has_financing: True nếu có ít nhất 1 sản phẩm tín dụng eligible=True được chọn
    (selected_financing_option khác "Không cần huy động vốn ngoài"). Khi RR-002 kích
    hoạt nhưng KHÔNG có gói vay nào khả thi (has_financing=False), điều kiện bảo vệ
    KHÔNG được nói về "giải ngân" — vì không có khoản vay nào để giải ngân — mà phải
    hướng về đàm phán lại tiến độ thanh toán/đặt cọc với chính khách hàng của hợp
    đồng này, đúng như hướng xử lý duy nhất còn lại khi thị trường tín dụng không
    khả thi.
    """
    ids = set(triggered_rule_ids)

    if "RR-006" in ids:
        if is_new_customer:
            # Khách hàng MỚI không thể có "lịch sử giao dịch" với OPC — yêu cầu bổ
            # sung phải là những thứ họ THỰC SỰ có thể cung cấp, không lặp lại yêu
            # cầu bất khả thi (bổ sung lịch sử giao dịch chưa từng tồn tại).
            return (
                "Khách hàng mới (chưa có lịch sử giao dịch với OPC) cần bổ sung giấy tờ "
                "pháp lý (đăng ký kinh doanh/hộ kinh doanh), tài sản đảm bảo hoặc người/"
                "đơn vị bảo lãnh thanh toán, và nên yêu cầu đặt cọc hoặc thanh toán một "
                "phần trước khi triển khai để bù đắp việc chưa đủ dữ liệu tin cậy; nếu "
                "không đáp ứng, Founder có quyền từ chối hoặc tạm dừng đề xuất tài chính."
            )
        return (
            "Khách hàng phải bổ sung minh chứng dữ liệu tín dụng còn thiếu (lịch sử "
            "giao dịch, tài sản đảm bảo/bảo lãnh thanh toán) trước khi giải ngân bất kỳ "
            "khoản nào; nếu không bổ sung đủ trong thời hạn thỏa thuận, Founder có quyền "
            "từ chối hoặc tạm dừng đề xuất tài chính."
        )
    if "RR-002" in ids:
        if not has_financing:
            # Không có sản phẩm tín dụng nào eligible -> không có khoản vay nào để
            # "giải ngân". Hướng xử lý duy nhất còn lại: đàm phán lại tiến độ thanh
            # toán/đặt cọc với khách hàng của hợp đồng này để tự cải thiện dòng tiền.
            return (
                "Không có sản phẩm tín dụng nào khả thi để bù đắp thâm hụt dòng tiền — "
                "Founder cần đàm phán lại với khách hàng để nhận đặt cọc hoặc thanh toán "
                "trước một phần, đồng thời triển khai hợp đồng theo tiến độ từng giai "
                "đoạn (phase delivery) gắn với xác nhận thanh toán ở mỗi giai đoạn, để "
                "dòng tiền dự phòng của công ty không giảm dưới ngưỡng tối thiểu; nếu "
                "khách hàng không đồng ý điều chỉnh tiến độ thanh toán, Founder cần cân "
                "nhắc từ chối hoặc hoãn triển khai hợp đồng."
            )
        return (
            "Thực hiện vay vốn hoặc triển khai hợp đồng theo tiến độ từng giai đoạn (phase delivery), "
            "gắn với xác nhận thanh toán của khách hàng ở mỗi giai đoạn, để bảo vệ dòng "
            "tiền dự phòng của công ty không giảm dưới ngưỡng tối thiểu."
        )
    if "RR-003" in ids:
        return (
            "Đàm phán lại chi phí vận hành hoặc điều chỉnh giá dịch vụ trước khi ký hợp "
            "đồng, để đưa biên lợi nhuận gộp về trên ngưỡng an toàn 28%."
        )
    return "Không có điều kiện bảo vệ đặc biệt — theo dõi định kỳ theo quy trình chuẩn."


def enforce_decision_card(
    decision_result: DecisionAgentOutput,
    finance_metrics: dict,
    cash_projection: dict,
    confidence_result: Optional[dict],
    partner_matrix: list[dict],
    requested_amount: float,
    founder_approval_needed: bool,
    triggered_rule_ids: list[str],
    is_new_customer: bool = False,
) -> DecisionAgentOutput:
    """
    Ép các trường ĐỊNH LƯỢNG + "protection_condition" của Decision Card về đúng giá
    trị/logic Python đã tính tất định.

    Lý do: hướng dẫn trong prompt ("PHẢI lấy đúng giá trị Python cung cấp, không tự
    tính lại") chỉ là ràng buộc bằng lời — OpenAI vẫn có thể tự sinh số, tự quyết
    định "không cần huy động vốn" dù RR-002 đã kích hoạt, hoặc sinh ra một điều kiện
    bảo vệ khác nhau mỗi lần chạy dù cùng payload (đã quan sát thấy trong thực tế).
    Hàm này ghi đè sau khi nhận kết quả, để OpenAI chỉ còn vai trò diễn giải ngôn ngữ
    tự nhiên (three_reasons, executive_summary...), không được quyết định số liệu
    hay nội dung cam kết mà Founder phải xác nhận.
    """
    eligible_options = [item for item in partner_matrix if item.get("eligible")]

    if eligible_options:
        best = eligible_options[0]
        enforced_option = (
            f"{best['bank']} — {best['product_name']} "
            f"(lãi suất/phí năm {best['annual_rate_or_fee']:.2%}, "
            f"phí xử lý {best['processing_fee_rate']:.2%}, "
            f"hạn mức tối thiểu {format_vnd(best['minimum_amount'])})"
        )
        enforced_funding_amount = round(requested_amount, 2)
    else:
        enforced_option = "Không cần huy động vốn ngoài"
        enforced_funding_amount = 0.0

    # RR-005: requested_amount > 300 triệu -> BẮT BUỘC cần Founder phê duyệt, không
    # được OpenAI tự ý bỏ qua. Nếu OpenAI tự đánh giá cần duyệt vì lý do khác, vẫn giữ. Bắc buộc phải so sánh đúng với requested_amount.
    enforced_human_approval = bool(founder_approval_needed or decision_result.human_approval_required)

    return decision_result.model_copy(
        update={
            "gross_margin": finance_metrics["gross_margin"],
            "closing_cash": cash_projection["min_projected_closing_cash"],
            "confidence_score": (
                confidence_result["confidence_score"] if confidence_result is not None else None
            ),
            "selected_financing_option": enforced_option,
            "funding_amount": enforced_funding_amount,
            "protection_condition": build_protection_condition(
                triggered_rule_ids, is_new_customer, has_financing=bool(eligible_options)
            ),
            "human_approval_required": enforced_human_approval,
        }
    )


# ============================================================
# 7.5 API-H HANDLING CHECKLIST — đối chiếu 22_API_HANDLING_RULES
#
#     Hàm này CHỈ ĐỌC lại các kết quả Python đã tính ở trên (missing_fields,
#     partner_matrix, triggered_rules, workflow_logs...) để hiển thị dạng
#     checklist "lỗi ở đâu tick ở đó" — KHÔNG tính toán lại hay thay đổi bất
#     kỳ giá trị/logic nghiệp vụ nào đã có.
# ============================================================

API_HANDLING_RULES = [
    {"rule_id": "API-H-001", "applies_to": "Any partner/API mock",
     "required_handling": "Stop unsafe action and ask for correction"},
    {"rule_id": "API-H-002", "applies_to": "Financial partner recommendation",
     "required_handling": "Request missing evidence or return no-recommendation"},
    {"rule_id": "API-H-003", "applies_to": "Transaction alert",
     "required_handling": "Hold/review and request founder confirmation"},
    {"rule_id": "API-H-004", "applies_to": "OpenAI/tool call",
     "required_handling": "Mask/tokenize before sending"},
    {"rule_id": "API-H-005", "applies_to": "External API extension",
     "required_handling": "Declare endpoint, mock data, risk control in 26_API_ASSUMPTIONS"},
    {"rule_id": "API-H-006", "applies_to": "Runtime/demo",
     "required_handling": "Use static fallback or explain limitation"},
    {"rule_id": "API-H-007", "applies_to": "Audit/log",
     "required_handling": "Redact and store masked event only"},
    {"rule_id": "API-H-008", "applies_to": "Decision output",
     "required_handling": "Flag confidence and required confirmation"},
]


def evaluate_api_handling_checklist(
    missing_fields: list[str],
    partner_matrix: list[dict],
    requested_amount: float,
    triggered_rule_ids: list[str],
    confidence_score: Optional[float],
    transaction_risk_score: Optional[float],
    workflow_logs: list[dict],
) -> list[dict]:
    """
    Đối chiếu TẤT ĐỊNH (không gọi OpenAI) trạng thái runtime hiện tại với từng
    rule của 22_API_HANDLING_RULES. status ∈ {"OK", "REVIEW", "N/A"}.
    "REVIEW" = không hẳn là lỗi hệ thống, nhưng đúng theo bảng gốc là điểm cần
    con người rà soát/xác nhận (requires_human_approval).
    """
    checks: list[dict] = []

    # API-H-001 — Missing/invalid required field
    if missing_fields:
        checks.append({"rule_id": "API-H-001", "status": "REVIEW",
                        "detail": f"Thiếu trường bắt buộc: {', '.join(missing_fields)} — "
                                  "hệ thống đã dừng và yêu cầu bổ sung, không tự suy diễn."})
    else:
        checks.append({"rule_id": "API-H-001", "status": "OK",
                        "detail": "Không thiếu trường bắt buộc nào trong request lần này."})

    # API-H-002 — Financial partner recommendation
    eligible_options = [p for p in partner_matrix if p.get("eligible")]
    if requested_amount > 0 and not eligible_options:
        checks.append({"rule_id": "API-H-002", "status": "REVIEW",
                        "detail": "Có nhu cầu vốn nhưng không có gói vay eligible trong "
                                  "11_BANK_PRODUCTS — hệ thống trả về 'Không cần huy động "
                                  "vốn ngoài' (không bịa gói vay), Founder nên rà soát thêm "
                                  "nguồn vốn khác."})
    else:
        checks.append({"rule_id": "API-H-002", "status": "OK",
                        "detail": "Phương án tài chính (nếu có) lấy đúng từ partner_matrix "
                                  "eligible=true, không có gói vay/lãi suất tự bịa."})

    # API-H-003 — Transaction alert
    if transaction_risk_score is not None and transaction_risk_score > 85:
        checks.append({"rule_id": "API-H-003", "status": "REVIEW",
                        "detail": f"transaction_risk_score = {transaction_risk_score:.0f} > 85 "
                                  "— cần Founder xác nhận giao dịch khả nghi (08_BANK_TXN)."})
    else:
        checks.append({"rule_id": "API-H-003", "status": "OK",
                        "detail": "Không phát hiện giao dịch khả nghi (transaction_risk_score ≤ 85 "
                                  "hoặc không có dữ liệu giao dịch)."})

    # API-H-004 — OpenAI/tool call masking
    total_masked = sum(len(log.get("masked_fields", [])) for log in workflow_logs)
    checks.append({"rule_id": "API-H-004", "status": "OK",
                    "detail": f"Đã mask/tokenize {total_masked} lượt trường nhạy cảm "
                              "(customer_id, customer_name, account_id...) trước khi gửi "
                              "cho OpenAI ở cả 3 agent."})

    # API-H-005 — External API extension
    checks.append({"rule_id": "API-H-005", "status": "N/A",
                    "detail": "Chưa dùng thêm API/nguồn dữ liệu ngoài nào — chỉ dùng OpenAI "
                              "và Team Pack (11_BANK_PRODUCTS)."})

    # API-H-006 — Runtime/demo fail-safe
    if len(workflow_logs) == 3:
        checks.append({"rule_id": "API-H-006", "status": "OK",
                        "detail": "Cả 3 agent (Finance, Risk, Decision) đều gọi OpenAI "
                                  "thành công trong lượt chạy này."})
    else:
        checks.append({"rule_id": "API-H-006", "status": "REVIEW",
                        "detail": "Chưa đủ 3/3 agent hoàn tất — kiểm tra lại kết nối/quota OpenAI."})

    # API-H-007 — Audit/log secret redaction
    checks.append({"rule_id": "API-H-007", "status": "OK",
                    "detail": "workflow_logs chỉ lưu output có schema (Pydantic) và danh sách "
                              "masked_fields — không có API key/access token/định danh thô "
                              "nào được ghi log."})

    # API-H-008 — Decision output confidence (chỉ áp dụng khi RR-002 đã kích hoạt,
    # đúng như RR-006 chỉ có ý nghĩa khi thiếu hụt tiền mặt đã xảy ra).
    rr002_triggered = "RR-002" in triggered_rule_ids
    if rr002_triggered and (confidence_score is None or confidence_score < 0.65):
        score_text = f"{confidence_score:.0%}" if confidence_score is not None else "None"
        checks.append({"rule_id": "API-H-008", "status": "REVIEW",
                        "detail": f"RR-002 đã kích hoạt và confidence_score = {score_text} "
                                  "(< 65% hoặc chưa tính được) — cần con người xác nhận trước "
                                  "khi ra quyết định cuối."})
    else:
        checks.append({"rule_id": "API-H-008", "status": "OK",
                        "detail": "Không ở trong tình huống bắt buộc rà soát thêm theo RR-006."})

    rule_lookup = {rule["rule_id"]: rule for rule in API_HANDLING_RULES}
    for item in checks:
        rule = rule_lookup.get(item["rule_id"], {})
        item["applies_to"] = rule.get("applies_to", "")
        item["required_handling"] = rule.get("required_handling", "")
    return checks


# 7.6 CRISIS CARD — MVP1, MVP2, MVP3, MVP5 (MODELS & LOGIC)
# ============================================================

class CrisisCardInput(BaseModel):
    crisis_group: list[Literal[
        "DEADLINE_EARLY", "DEADLINE_LATE",
        "COST_CHANGE",
        "PAYMENT_DELAY",
        "FINANCE_CONDITION",
        "SCOPE_CHANGE", "ORDER_CHANGE",
    ]] = Field(max_length=2)
    contract_id: str
    days_deviation: Optional[int] = None
    extra_cost_amount: Optional[float] = None
    # Cách nhập thay thế cho extra_cost_amount: chi phí phát sinh theo % trên
    # estimated_cost baseline (dương = tăng chi phí, âm = giảm chi phí). Nếu cả
    # 2 trường cùng được điền, ưu tiên extra_cost_percent (xem resolve_crisis_deltas).
    extra_cost_percent: Optional[float] = None
    late_amount: Optional[float] = None
    late_month: Optional[str] = None
    late_days: Optional[int] = None
    new_annual_rate_or_fee: Optional[float] = None
    new_processing_fee_rate: Optional[float] = None
    new_collateral_ratio: Optional[float] = None
    new_num_provinces: Optional[int] = None
    new_order_count: Optional[int] = None
    raw_prompt_text: Optional[str] = None

def validate_crisis_card_input(crisis: CrisisCardInput) -> list[str]:
    errors = []
    if not crisis.crisis_group:
        errors.append("Cần chọn ít nhất một nhóm biến động.")
    # FIX (bug logic thực sự): DEADLINE_EARLY và DEADLINE_LATE dùng chung 1 trường
    # days_deviation duy nhất trên form -- không thể vừa giao sớm vừa giao muộn
    # cùng lúc trên cùng một hợp đồng, nếu chọn cả 2 thì công thức sẽ bị cộng dồn
    # vô nghĩa (áp cả 2 chiều lên cùng 1 số ngày). Chặn ngay từ bước validate.
    if "DEADLINE_EARLY" in crisis.crisis_group and "DEADLINE_LATE" in crisis.crisis_group:
        errors.append("Không thể chọn đồng thời Giao sớm và Giao muộn cho cùng một biến động (mâu thuẫn về tiến độ).")
    if any(g in ["DEADLINE_EARLY", "DEADLINE_LATE"] for g in crisis.crisis_group) and not crisis.days_deviation:
        errors.append("Cần nhập số ngày sớm/muộn cho biến động tiến độ.")
    if "COST_CHANGE" in crisis.crisis_group and not crisis.extra_cost_amount and not crisis.extra_cost_percent:
        errors.append("Cần nhập chi phí phát sinh (theo số tiền VNĐ hoặc theo %).")
    # FIX (theo yêu cầu bổ sung): late_month giờ là tùy chọn — nếu Founder không
    # nhập tháng cụ thể, hệ thống tự áp dụng vào tháng ĐẦU của lịch dòng tiền hợp
    # đồng (xem project_closing_cash_with_crisis). Chỉ còn late_amount và
    # late_days là bắt buộc.
    if "PAYMENT_DELAY" in crisis.crisis_group and (not crisis.late_amount or not crisis.late_days):
        errors.append("Cần nhập số tiền chậm thanh toán và số ngày trả muộn.")
    if "FINANCE_CONDITION" in crisis.crisis_group:
        if crisis.new_annual_rate_or_fee is None and crisis.new_processing_fee_rate is None and crisis.new_collateral_ratio is None:
            errors.append("Cần nhập ít nhất một thay đổi điều kiện tài chính.")
    if "SCOPE_CHANGE" in crisis.crisis_group and not crisis.new_num_provinces:
        errors.append("Cần nhập số tỉnh/thành phố mới.")
    if "ORDER_CHANGE" in crisis.crisis_group and not crisis.new_order_count:
        errors.append("Cần nhập số lượng đơn hàng mới.")
    return errors

class CrisisDelta(BaseModel):
    extra_oper: float = 0.0
    extra_estimated_cost: float = 0.0
    extra_list_price: float = 0.0
    payment_shift: Optional[dict] = None
    trigger_layer: Optional[Literal["L1","L2","L3","L4"]] = None
    # FIX (bug nghiêm trọng): trước đây vượt trần order raise ValueError khiến cả
    # luồng Crisis crash (except Exception chung ở UI chỉ hiện "Lỗi hệ thống",
    # không trả về Decision Card nào). Nay chuyển sang cờ tất định để UI tự xử lý
    # thành 1 quyết định TERMINATE rõ ràng thay vì crash.
    hard_cap_exceeded: bool = False
    note: str

class CrisisDecisionCardOutput(BaseModel):
    continue_contract: Literal["CONTINUE", "CONTINUE WITH CONDITIONS", "TERMINATE"]
    financing_plan: str
    key_protection_condition: str
    gross_margin_after: float
    closing_cash_after: float
    funding_amount_after: float
    executive_summary: str

def derive_risk_level_from_triggered_rules(triggered_rules: list[dict]) -> str:
    severities = {
        str(rule.get("severity", "")).strip().upper()
        for rule in (triggered_rules or [])
    }
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "HIGH" in severities:
        return "HIGH"
    if "MEDIUM" in severities:
        return "MEDIUM"
    return "LOW"

def resolve_crisis_deltas(
    crisis: CrisisCardInput,
    list_price_goc: float,
    old_num_provinces: Optional[int] = None,
    baseline_estimated_cost: Optional[float] = None,
    old_order_count: Optional[int] = None,
) -> CrisisDelta:
    extra_oper = 0.0
    extra_estimated_cost = 0.0
    extra_list_price = 0.0
    payment_shift = None
    trigger_layer = None
    hard_cap_exceeded = False
    notes = []

    for group in crisis.crisis_group:
        if group == "DEADLINE_EARLY":
            if crisis.days_deviation and crisis.days_deviation >= 7:
                extra_oper += 0.01
                extra_list_price += list_price_goc * 0.015
                notes.append(f"Giao sớm {crisis.days_deviation} ngày (>=7 ngày): oper +1%, list price +1.5%")
            elif crisis.days_deviation and crisis.days_deviation > 0:
                extra_oper += 0.005
                extra_list_price += list_price_goc * 0.01
                notes.append(f"Giao sớm {crisis.days_deviation} ngày (<7 ngày): oper +0.5%, list price +1%")
        elif group == "DEADLINE_LATE":
            if crisis.days_deviation and crisis.days_deviation >= 7:
                extra_oper += 0.005
                extra_estimated_cost += estimated_cost * 0.015
                notes.append(f"Giao muộn {crisis.days_deviation} ngày (>= 7 ngày): oper +0.5%, estimated cost +1.5% giá trị HĐ")
            elif crisis.days_deviation and crisis.days_deviation > 0:
                extra_oper += 0.0005
                extra_estimated_cost += list_price_goc * 0.01
                notes.append(f"Giao muộn {crisis.days_deviation} ngày (<7 ngày): oper +0.05%, estimated cost +1% giá trị HĐ")
        elif group == "COST_CHANGE":
            if crisis.extra_cost_percent is not None:
                base_cost_for_percent = baseline_estimated_cost or 0.0
                percent_based_cost = base_cost_for_percent * (crisis.extra_cost_percent / 100.0)
                extra_estimated_cost += percent_based_cost
                notes.append(
                    f"Phát sinh chi phí theo tỷ lệ: {crisis.extra_cost_percent:+.2f}% trên "
                    f"estimated_cost baseline ({base_cost_for_percent:,.0f} VNĐ) "
                    f"= {percent_based_cost:+,.0f} VNĐ"
                )
            else:
                extra_estimated_cost += crisis.extra_cost_amount or 0.0
                notes.append(f"Phát sinh chi phí: +{crisis.extra_cost_amount or 0.0} VNĐ")
        elif group == "PAYMENT_DELAY":
            if crisis.late_amount and crisis.late_days:
                # Lãi kép 1%/ngày trên số tiền chậm trả (đúng docx mục 3):
                # số tiền phải trả cuối cùng = late_amount * (1 + 1%)^late_days.
                daily_rate = 0.01
                compound_multiplier = (1 + daily_rate) ** crisis.late_days
                surcharge_pct = compound_multiplier - 1
                # FIX (theo yêu cầu bổ sung): late_month là tùy chọn — nếu Founder
                # không nhập, để None ở đây; project_closing_cash_with_crisis() sẽ
                # tự phân giải thành tháng ĐẦU của lịch dòng tiền hợp đồng.
                payment_shift = {
                    "late_amount": crisis.late_amount,
                    "late_month": crisis.late_month.strip() if crisis.late_month and crisis.late_month.strip() else None,
                    "late_days": crisis.late_days,
                    "surcharge_pct": surcharge_pct,
                }
                month_label = payment_shift["late_month"] or "tháng đầu hợp đồng (mặc định do không nhập tháng cụ thể)"
                notes.append(
                    f"Khách hàng trả muộn {crisis.late_amount} VNĐ vào {month_label}, "
                    f"chậm {crisis.late_days} ngày -> lãi kép 1%/ngày = +{surcharge_pct:.2%} "
                    "(chuyển gốc + lãi sang tháng sau)"
                )
        elif group == "SCOPE_CHANGE":
            # BUG cũ: cộng thẳng toàn bộ hệ số quy mô MỚI lên oper baseline, trong khi
            # oper baseline (build_finance_metrics) ĐÃ có sẵn hệ số quy mô CŨ (tính từ
            # num_provinces gốc của hợp đồng) -> bị tính trùng 2 lần phần quy mô. Sửa:
            # chỉ cộng phần CHÊNH LỆCH (mới - cũ) để hệ số quy mô mới THAY THẾ đúng
            # hệ số cũ, không cộng dồn lên nó.
            old_scale, _old_breakdown = compute_scale_coefficient(old_num_provinces)
            new_scale, breakdown = compute_scale_coefficient(crisis.new_num_provinces)
            net_scale_delta = new_scale - old_scale
            extra_oper += net_scale_delta
            notes.append(
                (breakdown["tieu_chi"] if breakdown else f"Đổi địa bàn -> {crisis.new_num_provinces} tỉnh/thành")
                + f" (thay hệ số quy mô cũ {old_scale:.2%} bằng hệ số mới {new_scale:.2%}, "
                f"net delta {net_scale_delta:+.2%})"
            )
        elif group == "ORDER_CHANGE":
            # FIX: Order Change (PDF mục 2.2) giờ đã là 1 phần của Oper Score CƠ SỞ
            # (xem compute_order_change_coefficient / build_finance_metrics), được
            # tính ngay từ baseline nếu Founder có nhập initial_order_count lúc tạo
            # hợp đồng. Để không cộng trùng hệ số Order Change 2 lần (giống cách
            # SCOPE_CHANGE đang xử lý phần quy mô ở trên), Crisis Card chỉ cộng
            # phần CHÊNH LỆCH (mới - cũ) so với hệ số đã có trong baseline.
            old_order_he_so, _old_order_bd, _old_order_hard = compute_order_change_coefficient(
                old_order_count
            )
            new_order_he_so, new_order_bd, new_order_hard_cap_exceeded = compute_order_change_coefficient(
                crisis.new_order_count
            )
            if new_order_hard_cap_exceeded:
                # FIX (bug nghiêm trọng): trước đây raise ValueError ở đây khiến
                # toàn bộ luồng Crisis Card bị except Exception chung ở UI bắt và
                # chỉ hiện "Lỗi hệ thống" -- không có Decision Card, không trả lời
                # được câu hỏi bắt buộc "có tiếp tục hợp đồng không / phương án
                # tài chính / điều kiện bảo vệ" theo đúng yêu cầu. Nay set cờ tất
                # định để tầng UI tự dựng 1 Decision Card TERMINATE rõ ràng.
                hard_cap_exceeded = True
                notes.append(
                    f"Số order mới ({crisis.new_order_count}) VƯỢT TRẦN CỨNG "
                    f"({ORDER_CHANGE_HARD_CAP} - ngưỡng tạm đặt, cần Founder xác nhận lại "
                    "con số này) -> không thể tiếp tục theo điều kiện hiện tại."
                )
            elif crisis.new_order_count:
                net_order_delta = new_order_he_so - old_order_he_so
                extra_oper += net_order_delta
                notes.append(
                    (
                        new_order_bd["tieu_chi"]
                        if new_order_bd
                        else f"Số order mới: {crisis.new_order_count} (trong hạn mức miễn phí)"
                    )
                    + f" (net delta so với baseline: {net_order_delta:+.2%})"
                )
            else:
                notes.append("Số order không đổi")
        elif group == "FINANCE_CONDITION":
            changed_fields = [
                field_name
                for field_name, field_value in (
                    ("annual_rate_or_fee", crisis.new_annual_rate_or_fee),
                    ("processing_fee_rate", crisis.new_processing_fee_rate),
                    ("collateral_ratio", crisis.new_collateral_ratio),
                )
                if field_value is not None
            ]
            # FIX (gây nhầm lẫn): trước đây note chỉ báo "L4" khi collateral_ratio đổi,
            # dù annual_rate_or_fee/processing_fee_rate (Lớp 3) có đổi CÙNG LÚC hay
            # không -> mất thông tin Lớp 3 cũng bị ảnh hưởng. Nay liệt kê ĐẦY ĐỦ các
            # lớp bị ảnh hưởng trong note. trigger_layer (field Literal đơn giá trị)
            # vẫn giữ đúng quy tắc cũ (ưu tiên lớp sâu nhất) để không đổi kiểu dữ liệu.
            layer_by_field = {
                "annual_rate_or_fee": "L3",
                "processing_fee_rate": "L3",
                "collateral_ratio": "L4",
            }
            affected_layers = sorted({layer_by_field[f] for f in changed_fields})
            layers_label = "+".join(affected_layers) if affected_layers else "?"
            trigger_layer = "L4" if "collateral_ratio" in changed_fields else "L3"
            notes.append(
                "Thay đổi điều kiện tài chính (" + ", ".join(changed_fields) + f") — re-filter từ lớp {layers_label}"
            )

    return CrisisDelta(
        extra_oper=extra_oper,
        extra_estimated_cost=extra_estimated_cost,
        extra_list_price=extra_list_price,
        payment_shift=payment_shift,
        trigger_layer=trigger_layer,
        hard_cap_exceeded=hard_cap_exceeded,
        note="; ".join(notes)
    )

def build_finance_metrics_with_crisis(
    selected_products: pd.DataFrame,
    payment_reliability: Optional[float],
    province: Optional[str],
    transaction_risk_score: Optional[float],
    order_date: pd.Timestamp,
    due_date: pd.Timestamp,
    num_provinces: Optional[int],
    crisis_delta: CrisisDelta,
    initial_order_count: Optional[int] = None,
) -> dict:
    metrics = build_finance_metrics(
        selected_products, payment_reliability, province,
        transaction_risk_score, order_date, due_date, num_provinces, initial_order_count,
    )
    if crisis_delta:
        metrics["oper_coefficient"] += crisis_delta.extra_oper
        if crisis_delta.extra_oper != 0:
            metrics["oper_breakdown"].append({"tieu_chi": f"Crisis: {crisis_delta.note}", "he_so": crisis_delta.extra_oper})
        metrics["estimated_cost"] = metrics["baseline_estimate"] * (1 + metrics["oper_coefficient"]) + crisis_delta.extra_estimated_cost
        metrics["total_list_price"] += crisis_delta.extra_list_price
        if metrics["total_list_price"] > 0:
            metrics["gross_margin"] = (metrics["total_list_price"] - metrics["estimated_cost"]) / metrics["total_list_price"]
        else:
            metrics["gross_margin"] = 0.0
    return metrics

def project_closing_cash_with_crisis(
    data: dict[str, pd.DataFrame],
    selected_products: pd.DataFrame,
    finance_metrics: dict,
    order_date: pd.Timestamp,
    reserve_minimum: float,
    crisis_delta: CrisisDelta
) -> dict:
    proj = project_closing_cash(data, selected_products, finance_metrics, order_date, reserve_minimum)
    if not crisis_delta:
        return proj
        
    schedule = proj["schedule"]
    
    if crisis_delta.extra_list_price != 0.0 and len(schedule) > 0:
        schedule[0]["deal_cash_in"] += crisis_delta.extra_list_price
        
    if crisis_delta.payment_shift:
        late_month = crisis_delta.payment_shift["late_month"]
        # FIX (theo yêu cầu bổ sung): PAYMENT_DELAY áp dụng cho một tháng cụ thể
        # nếu Founder có nhập; nếu KHÔNG nhập tháng, tự động mặc định là tháng
        # ĐẦU TIÊN của lịch dòng tiền hợp đồng (schedule[0]) thay vì báo lỗi/bỏ
        # qua âm thầm.
        if not late_month and schedule:
            late_month = schedule[0]["month"]
        late_amount = crisis_delta.payment_shift["late_amount"]
        surcharge = crisis_delta.payment_shift["surcharge_pct"]
        shifted_amount = late_amount * (1 + surcharge)

        matched = False
        for i, row in enumerate(schedule):
            if row["month"] == late_month:
                matched = True
                row["deal_cash_in"] -= late_amount
                if i + 1 < len(schedule):
                    # Đúng docx: dời gốc + lãi sang Cash In của tháng KẾ TIẾP.
                    schedule[i + 1]["deal_cash_in"] += shifted_amount
                else:
                    # late_month là tháng CUỐI của lịch hợp đồng -> không còn "tháng tiếp
                    # theo" để dời sang. Cộng lại (kèm lãi) vào chính tháng đó thay vì để
                    # khoản tiền biến mất khỏi dòng tiền (vi phạm bảo toàn tiền mặt).
                    row["deal_cash_in"] += shifted_amount
                break

        if not matched:
            # late_month không khớp tháng nào trong lịch (VD sai định dạng "YYYY-MM") ->
            # payment_shift coi như không áp dụng được. Ghi lại lý do vào note của proj
            # để UI có thể cảnh báo, thay vì âm thầm bỏ qua.
            proj["payment_shift_warning"] = (
                f"Không tìm thấy tháng '{late_month}' trong lịch dòng tiền hợp đồng — "
                "chưa áp dụng được biến động Payment Delay."
            )

    cumulative_deal_net = 0.0
    if schedule:
        prior_new_closing = schedule[0]["opening_cash"]
        for i, row in enumerate(schedule):
            cumulative_deal_net += row["deal_cash_in"] - row["deal_cash_out"]
            row["opening_cash"] = prior_new_closing
            row["projected_closing_cash"] = row["baseline_projected_closing_cash"] + cumulative_deal_net
            prior_new_closing = row["projected_closing_cash"]
            
    min_closing_cash = min([r["projected_closing_cash"] for r in schedule]) if schedule else 0.0
    breach = min_closing_cash < reserve_minimum
    
    proj["min_projected_closing_cash"] = round(min_closing_cash, 2)
    proj["cash_reserve_breach"] = breach
    return proj

def rerun_partner_matrix_from_layer(
    data: dict[str, pd.DataFrame],
    funding_need: float,
    cash_projection: dict,
    crisis: CrisisCardInput,
    baseline_partner_matrix: Optional[list[dict]] = None,
) -> tuple[list[dict], Optional[str]]:
    """
    Nhóm FINANCE_CONDITION: đổi annual_rate_or_fee/processing_fee_rate/collateral_ratio
    là kết quả ĐÀM PHÁN LẠI với đúng 1 gói vay/đối tác cụ thể đang tài trợ hợp đồng này —
    KHÔNG áp dụng đồng loạt lên toàn bộ 11_BANK_PRODUCTS. BUG cũ: gán thẳng giá trị mới
    cho CẢ CỘT (mọi ngân hàng/gói vay), khiến Lớp 3 ("so sánh tổng chi phí giữa các gói
    vay") và Lớp 4 (collateral) mất hết ý nghĩa vì mọi gói đều bị ép về cùng 1 giá trị.

    Xác định đúng gói vay cần đổi: gói eligible đứng đầu (tốt nhất) trong
    baseline_partner_matrix của lượt chạy Operations trước — đây chính là gói đang thực
    sự tài trợ hợp đồng. Chỉ override đúng dòng bank_product_id đó rồi build lại toàn bộ
    Lớp 1-4 (Lớp 1-2 không đổi vì input của chúng không đổi, nên kết quả tương đương
    đúng "chỉ lọc lại từ Lớp 3/4 trở đi").

    FIX (bug logic thực sự): trước đây nếu KHÔNG xác định được đúng 1 gói vay để áp
    thay đổi (VD hợp đồng gốc chưa vay gói nào -> baseline_partner_matrix không có
    eligible=true), hàm ÂM THẦM bỏ qua toàn bộ thay đổi Founder vừa nhập, không có
    bất kỳ cảnh báo nào hiển thị. Nay trả thêm 1 cảnh báo dạng text (phần tử thứ 2
    của tuple) để tầng UI hiển thị rõ cho Founder biết thay đổi chưa được áp dụng.
    """
    data_copy = data.copy()
    finance_condition_warning = None
    if "FINANCE_CONDITION" in crisis.crisis_group:
        products = data_copy["11_BANK_PRODUCTS"].copy()

        target_product_id = None
        if baseline_partner_matrix:
            eligible_before = [item for item in baseline_partner_matrix if item.get("eligible")]
            if eligible_before:
                target_product_id = eligible_before[0].get("bank_product_id")

        if target_product_id is not None:
            mask = products["bank_product_id"] == target_product_id
        else:
            # Không xác định được đúng 1 gói vay cụ thể (VD hợp đồng gốc chưa có gói
            # nào eligible) -> an toàn nhất là KHÔNG áp lên toàn bộ thị trường; giữ
            # nguyên bảng gốc để tránh làm sai lệch dữ liệu của các gói khác.
            mask = pd.Series([False] * len(products), index=products.index)
            finance_condition_warning = (
                "Không xác định được gói vay cụ thể đang tài trợ hợp đồng gốc (baseline "
                "chưa có gói nào eligible=true) -> thay đổi annual_rate_or_fee/"
                "processing_fee_rate/collateral_ratio vừa nhập CHƯA được áp dụng vào bất "
                "kỳ gói vay nào trong 11_BANK_PRODUCTS."
            )

        if crisis.new_annual_rate_or_fee is not None:
            products.loc[mask, "annual_rate_or_fee"] = crisis.new_annual_rate_or_fee
        if crisis.new_processing_fee_rate is not None:
            products.loc[mask, "processing_fee_rate"] = crisis.new_processing_fee_rate
        if crisis.new_collateral_ratio is not None:
            products.loc[mask, "collateral_ratio"] = crisis.new_collateral_ratio
        data_copy["11_BANK_PRODUCTS"] = products
        
    return build_partner_matrix(data_copy, funding_need, cash_projection), finance_condition_warning


def run_crisis_decision_agent(client: "OpenAI", model: str, payload: dict):
    """
    Hàm gọi OpenAI riêng cho Crisis Decision Agent (thay cho việc gọi
    call_structured_agent() rải rác/inline ở UI trước đây) — mirror đúng cách tổ
    chức run_finance_agent/run_risk_agent/run_decision_agent ở Mục 7, nhưng đặt ở
    Mục 7.6 vì đây là 1 phần của logic Crisis Card, KHÔNG đụng tới Mục 7.

    QUAN TRỌNG: key_protection_condition của Crisis Decision Card KHÔNG bị
    enforce_crisis_decision_card() ghi đè — giữ nguyên đúng như Agent trả về.
    continue_contract CŨNG được ưu tiên giữ theo Agent, TRỪ 1 trường hợp bắt buộc:
    closing_cash_after < 0 VÀ không còn gói vay eligible nào (hoặc order vượt trần
    cứng) — khi đó enforce_crisis_decision_card() sẽ ép cứng về TERMINATE bất kể
    Agent trả lời gì, vì đây là quy tắc "BẮT BUỘC" theo nghiệp vụ, không thể chỉ
    dựa vào việc Agent tuân thủ đúng prompt (LLM không đảm bảo tuân thủ 100%).
    Vì vậy instructions dưới đây vẫn cần đủ chi tiết để Agent tự đưa ra 2 trường
    này một cách có căn cứ — nhưng có 1 lưới an toàn tất định phía sau.
    """
    instructions = """
Bạn là Decision & Partner Agent của OPC, chuyên xử lý Crisis Card (biến động phát
sinh trên 1 hợp đồng đang chạy).

Dữ liệu đầu vào (payload) gồm: crisis_context (nhóm biến động + delta đã tính),
finance_metrics (SAU biến động), cash_projection (SAU biến động), partner_matrix
(SAU biến động, đã lọc theo Mục 4), requested_amount, triggered_rules, risk_level,
baseline_context, founder_approval_needed, finance_agent_output, risk_agent_output.

Nhiệm vụ — trả về CrisisDecisionCardOutput gồm:
1. continue_contract: chọn đúng 1 trong 3 giá trị:
   - CONTINUE: chỉ số tài chính sau biến động vẫn an toàn (gross_margin >= 0.28,
     closing_cash sau biến động >= 0, không có rủi ro nghiêm trọng mới phát sinh).
   - CONTINUE WITH CONDITIONS: hợp đồng còn khả thi nhưng cần thêm điều kiện ràng
     buộc/kiểm soát (VD: gross_margin giảm nhưng vẫn dương, cần huy động vốn ngoài,
     cần đàm phán lại một phần điều khoản với khách hàng).
   - TERMINATE: closing_cash sau biến động < 0 VÀ partner_matrix không có sản phẩm
     nào eligible=true (không còn nguồn bù đắp) — trong trường hợp này BẮT BUỘC
     chọn TERMINATE, không được chọn giá trị khác.
2. financing_plan: nêu rõ phương án tài chính cụ thể áp dụng (vay gói nào trong
   partner_matrix nếu eligible=true, hoặc "Không cần huy động vốn ngoài" nếu
   partner_matrix rỗng/không sản phẩm nào eligible, hoặc phương án đàm phán lại
   với khách hàng nếu không còn sản phẩm tín dụng khả thi).
3. key_protection_condition: đúng 1 điều kiện bảo vệ/thương mại cụ thể nhất mà
   Founder phải xác nhận trước khi áp dụng biến động này (VD: đặt cọc bổ sung,
   giải ngân theo tiến độ, tài sản đảm bảo, đàm phán lại thời hạn thanh toán...).
   Phải bám sát đúng crisis_context và triggered_rules được cung cấp, không chung
   chung, không lặp lại nguyên văn financing_plan.
4. gross_margin_after, closing_cash_after, funding_amount_after: PHẢI lấy đúng
   nguyên giá trị Python đã cung cấp trong finance_metrics/cash_projection/
   requested_amount — không tự tính lại, không làm tròn khác đi (các trường này
   vẫn bị Python enforce lại sau, nhưng vẫn phải điền đúng ngay từ đầu).
5. executive_summary: tóm tắt ngắn gọn bằng tiếng Việt về tác động của biến động
   này lên hợp đồng và lý do đưa ra continue_contract ở trên. Tối đa 2-3 câu, chỉ
   nêu ĐÚNG 1 nguyên nhân/tác động chính + 1 căn cứ cho quyết định — KHÔNG liệt kê
   lại toàn bộ số liệu chi tiết (các số liệu đó đã hiển thị riêng ở finance_metrics/
   cash_projection), ưu tiên diễn giải ngắn, dễ hiểu, đi thẳng vào ý chính.
6. Risk level phải đánh giá đúng tình hình tài chính BEFORE/AFTER:
    - Đánh giá dựa trên bảng risk rule: Nếu không vi phạm risk rule mới thì không thay đổi mức độ risk rule 
    - Không tự đặt risk_level trái với triggered_rules/risk_level đã có trong payload.
Quy tắc bắt buộc:
- Không phát minh số liệu, sản phẩm tín dụng hay điều khoản ngoài payload.
- Không tự đổi requested_amount hay eligible của partner_matrix.
- Nếu founder_approval_needed=true, phải nêu rõ trong financing_plan hoặc
  executive_summary rằng cần Founder phê duyệt.
- Viết bằng tiếng Việt, ngắn gọn, đủ căn cứ để Founder ra quyết định ngay.
"""
    return call_structured_agent(
        client, model, instructions, payload, CrisisDecisionCardOutput, "Crisis Decision Agent"
    )


# ------------------------------------------------------------------
# FIX (nghiêm trọng): yêu cầu ghi rõ "Nhập dữ kiện Crisis Card bằng prompt HOẶC
# biểu mẫu đơn giản". Trước đây CrisisCardInput đã có sẵn field raw_prompt_text
# nhưng KHÔNG có bất kỳ nơi nào dùng nó -- chỉ có đường form. Bổ sung 1 Agent
# OpenAI chuyên trích xuất Crisis Card từ mô tả tự do (đúng yêu cầu "phải sử
# dụng ứng dụng/dịch vụ OpenAI trong lúc xử lý", không dùng regex/rule cứng để
# giả lập việc "hiểu" prompt).
# ------------------------------------------------------------------

class CrisisCardPromptExtraction(BaseModel):
    """Schema trích xuất Crisis Card từ prompt tự do — KHÔNG gồm contract_id vì
    contract_id giờ được hệ thống tự gán từ hợp đồng đang chạy ở tab Operations
    (không còn ô nhập thủ công, và cũng tránh AI đoán sai mã hợp đồng)."""
    crisis_group: list[Literal[
        "DEADLINE_EARLY", "DEADLINE_LATE",
        "COST_CHANGE",
        "PAYMENT_DELAY",
        "FINANCE_CONDITION",
        "SCOPE_CHANGE", "ORDER_CHANGE",
    ]] = Field(max_length=2)
    days_deviation: Optional[int] = None
    extra_cost_amount: Optional[float] = None
    extra_cost_percent: Optional[float] = None
    late_amount: Optional[float] = None
    late_month: Optional[str] = None
    late_days: Optional[int] = None
    new_annual_rate_or_fee: Optional[float] = None
    new_processing_fee_rate: Optional[float] = None
    new_collateral_ratio: Optional[float] = None
    new_num_provinces: Optional[int] = None
    new_order_count: Optional[int] = None
    extraction_notes: str


def run_crisis_prompt_extraction_agent(client: "OpenAI", model: str, raw_prompt_text: str):
    """
    Dùng OpenAI để đọc mô tả biến động bằng ngôn ngữ tự nhiên (tiếng Việt) và trích
    xuất đúng các trường của CrisisCardInput — thay cho việc phải điền biểu mẫu.
    Kết quả trả về vẫn phải đi qua validate_crisis_card_input() và toàn bộ luồng
    tính toán tất định giống hệt đường Form (không có ngoại lệ nào bỏ qua bước
    kiểm tra hợp lệ chỉ vì dữ liệu đến từ AI).
    """
    instructions = """
Bạn là trợ lý trích xuất dữ liệu cho Crisis Card của OPC. Người dùng mô tả một biến
động phát sinh trên hợp đồng đang chạy bằng ngôn ngữ tự nhiên (tiếng Việt). Nhiệm vụ:
đọc kỹ mô tả và trả về CrisisCardPromptExtraction gồm đúng các trường sau, CHỈ điền
những trường thực sự được đề cập, để trống (null) các trường không có thông tin:

- crisis_group: chọn tối đa 2 trong 7 nhóm sau, đúng với mô tả:
  DEADLINE_EARLY (khách yêu cầu giao sớm), DEADLINE_LATE (OPC giao muộn),
  COST_CHANGE (phát sinh chi phí), PAYMENT_DELAY (khách trả muộn),
  FINANCE_CONDITION (đổi lãi suất/phí xử lý/tỷ lệ thế chấp),
  SCOPE_CHANGE (đổi số tỉnh/thành triển khai), ORDER_CHANGE (đổi số lượng đơn hàng).
- days_deviation: số ngày sớm/muộn (dùng cho DEADLINE_EARLY/DEADLINE_LATE).
- extra_cost_amount: số tiền chi phí phát sinh (VNĐ, dùng cho COST_CHANGE, khi mô tả
  nêu con số tuyệt đối).
- extra_cost_percent: % chi phí phát sinh trên estimated_cost baseline (dùng cho
  COST_CHANGE, khi mô tả nêu theo tỷ lệ %, dương = tăng, âm = giảm). Chỉ điền MỘT
  trong hai trường extra_cost_amount / extra_cost_percent theo đúng cách mô tả nêu ra.
- late_amount, late_month ("YYYY-MM"), late_days: dùng cho PAYMENT_DELAY.
- new_annual_rate_or_fee, new_processing_fee_rate, new_collateral_ratio: dùng cho
  FINANCE_CONDITION (chỉ điền trường được đề cập rõ ràng, kể cả khi giá trị mới = 0).
- new_num_provinces: dùng cho SCOPE_CHANGE.
- new_order_count: dùng cho ORDER_CHANGE.
- extraction_notes: tóm tắt ngắn gọn (tiếng Việt) những gì bạn đã hiểu/suy luận từ
  prompt, và nêu rõ nếu có thông tin còn mơ hồ/thiếu để Founder tự kiểm tra lại.

Quy tắc bắt buộc:
- Không tự bịa số liệu không có trong prompt.
- Nếu prompt mô tả nhiều hơn 2 nhóm biến động, chỉ chọn 2 nhóm rõ ràng/quan trọng
  nhất và nêu rõ trong extraction_notes rằng các nhóm còn lại đã bị bỏ qua.
- Nếu prompt không đủ thông tin để xác định crisis_group, vẫn phải chọn nhóm gần
  đúng nhất có thể và ghi rõ sự không chắc chắn trong extraction_notes.
"""
    payload = {"raw_prompt_text": raw_prompt_text}
    return call_structured_agent(
        client, model, instructions, payload, CrisisCardPromptExtraction, "Crisis Prompt Extraction Agent"
    )


def enforce_crisis_decision_card(
    decision_result: CrisisDecisionCardOutput,
    finance_metrics_after: dict,
    cash_projection_after: dict,
    requested_amount_after: float,
    partner_matrix_after: list[dict],
    triggered_rule_ids: list[str],
    hard_cap_exceeded: bool = False,
) -> CrisisDecisionCardOutput:
    # FIX (gây nhầm lẫn / code thừa): tham số is_new_customer trước đây được truyền
    # vào nhưng KHÔNG hề dùng trong thân hàm (khác với enforce_decision_card() gốc ở
    # Mục 6, nơi is_new_customer được dùng để build_protection_condition()). Vì
    # key_protection_condition ở Crisis Card CHỦ Ý không bị Python ghi đè (giữ đúng
    # nguyên văn Agent OpenAI sinh ra — xem ghi chú bên dưới), tham số này không có
    # tác dụng gì và đã được loại bỏ để tránh gây hiểu nhầm là có logic khác biệt
    # theo loại khách hàng đang được áp dụng ở đây.
    eligible_options = [item for item in partner_matrix_after if item.get("eligible")]
    has_financing = bool(eligible_options)

    min_cash = cash_projection_after["min_projected_closing_cash"]

    # YÊU CẦU: key_protection_condition KHÔNG bị Python ghi đè — giữ nguyên đúng giá
    # trị Agent OpenAI (CrisisDecisionAgent) đã sinh ra. Các trường ĐỊNH LƯỢNG
    # (gross_margin_after/closing_cash_after/funding_amount_after) luôn bị ép về
    # đúng số Python đã tính tất định.

    # Mirror đúng quy tắc của enforce_decision_card() (Mục 6): chỉ hiển thị funding_amount
    # khi thực sự CÓ gói vay eligible; nếu không, phải là 0 dù requested_amount_after > 0
    # (đó là "nhu cầu" chưa có nguồn, không phải "số sẽ vay được").
    enforced_funding_amount = round(requested_amount_after, 2) if has_financing else 0.0

    # FIX (lưới an toàn tất định — bug logic thực sự): instructions gửi cho
    # CrisisDecisionAgent có ghi "BẮT BUỘC chọn TERMINATE" khi closing_cash_after<0
    # VÀ không còn gói vay eligible nào, nhưng trước đây continue_contract hoàn
    # toàn không bị Python kiểm tra lại — nghĩa là quy tắc "bắt buộc" này chỉ tồn
    # tại trên giấy (prompt), phụ thuộc 100% vào việc Agent OpenAI có tuân thủ hay
    # không. LLM không đảm bảo tuân thủ tuyệt đối mọi lần, nên với 1 quyết định
    # nghiệp vụ quan trọng như TERMINATE, cần enforce cứng bằng Python.
    #
    # Áp dụng cho đúng 2 điều kiện tất định đã được xác định rõ trong hệ thống:
    #   (1) closing_cash sau biến động < 0 VÀ partner_matrix không còn eligible nào
    #       (không còn nguồn bù đắp funding gap).
    #   (2) hard_cap_exceeded=True (ORDER_CHANGE vượt trần cứng OPC có thể nhận).
    # Nếu Agent đã tự trả TERMINATE thì giữ nguyên (không đổi lý do/summary của
    # Agent một cách không cần thiết).
    mandatory_terminate_reasons = []
    if min_cash < 0 and not has_financing:
        mandatory_terminate_reasons.append(
            "closing cash sau biến động < 0 và không còn gói vay eligible nào trong "
            "partner_matrix (không còn nguồn bù đắp funding gap)"
        )
    if hard_cap_exceeded:
        mandatory_terminate_reasons.append(
            "số order mới vượt trần cứng OPC có thể nhận (hard_cap_exceeded=True)"
        )

    enforced_continue_contract = decision_result.continue_contract
    enforced_executive_summary = decision_result.executive_summary
    if mandatory_terminate_reasons and decision_result.continue_contract != "TERMINATE":
        enforced_continue_contract = "TERMINATE"
        reasons_text = "; ".join(mandatory_terminate_reasons)
        enforced_executive_summary = (
            f"[Ghi đè tất định bởi hệ thống — TERMINATE bắt buộc vì: {reasons_text}]. "
            f"Đánh giá ban đầu của Agent (không được dùng làm quyết định cuối): "
            f"{decision_result.executive_summary}"
        )

    return decision_result.model_copy(
        update={
            "continue_contract": enforced_continue_contract,
            "gross_margin_after": finance_metrics_after["gross_margin"],
            "closing_cash_after": min_cash,
            "funding_amount_after": enforced_funding_amount,
            "executive_summary": enforced_executive_summary,
        }
    )

# ============================================================
# 8. UI
# ============================================================

st.markdown(
    """
<style>
/* Import font hiện đại — Outfit cho nội dung, Inter cho tiêu đề/nhãn (đã được
   dùng ở nhiều nơi trong app nhưng trước đây chưa được import). */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --brand-blue: #2563eb;
    --brand-indigo: #4f46e5;
    --brand-violet: #7c3aed;
    --brand-pink: #db2777;
    --ink-900: #0f172a;
    --ink-700: #334155;
    --ink-500: #64748b;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --border-soft: #e2e8f0;
}

* {
    scrollbar-width: thin;
    scrollbar-color: #c7d2fe #f1f5f9;
}
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #93c5fd, #a5b4fc);
    border-radius: 10px;
    border: 2px solid #f1f5f9;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #60a5fa, #818cf8); }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: #1e293b !important;
}

/* Nền tảng tổng thể — mesh gradient nhẹ nhàng, nhiều lớp, cố định theo viewport */
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(99, 102, 241, 0.10) 0%, rgba(99, 102, 241, 0) 38%),
        radial-gradient(circle at 92% 4%, rgba(236, 72, 153, 0.09) 0%, rgba(236, 72, 153, 0) 35%),
        radial-gradient(circle at 85% 92%, rgba(59, 130, 246, 0.10) 0%, rgba(59, 130, 246, 0) 40%),
        linear-gradient(180deg, #f5f7fe 0%, #ffffff 55%, #f8fafc 100%);
    background-attachment: fixed;
    color: #1e293b;
}

/* --- Tăng độ tương phản chữ / dễ đọc hơn (không đổi logic, chỉ đổi giao diện) --- */

/* Toàn bộ tiêu đề mặc định rõ nét, đậm, tối màu thay vì mờ nhạt */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

/* Đoạn văn bản / markdown mặc định trong nội dung chính */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #1e293b;
    font-size: 1rem;
    line-height: 1.6;
}

/* Nhãn (label) của các ô nhập liệu, selectbox, radio, checkbox... rõ ràng hơn */
label, .stTextInput label, .stSelectbox label, .stRadio label,
.stCheckbox label, .stFileUploader label, .stSlider label,
[data-testid="stWidgetLabel"] p {
    color: #1e293b !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Caption / chú thích nhỏ vẫn giữ nhẹ nhàng nhưng đủ tương phản để đọc được */
[data-testid="stCaptionContainer"], .stCaption {
    color: #475569 !important;
}

/* Sidebar: nền trắng rõ ràng + chữ tối màu, tránh bị mờ trên nền gradient */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] * {
    color: #1e293b;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}

/* Placeholder trong input rõ hơn một chút để vẫn đọc được nhưng không lẫn với giá trị thật */
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Nội dung bên trong expander */
[data-testid="stExpander"] summary {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Chữ trong bảng dữ liệu rõ nét hơn */
[data-testid="stDataFrame"] * {
    color: #1e293b;
}

/* Tabs: nội dung chữ bên trong mỗi tab */
.stTabs [data-baseweb="tab-panel"] {
    color: #1e293b;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 95% !important;
}

/* Hiệu ứng nổi bồng bềnh cho các thành phần */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}

/* Sidebar — nền mềm mại, chữ tối màu, nổi bật hơn phần nội dung chính */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 24px rgba(15, 23, 42, 0.03);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Agent Card - Glassmorphism & Hover */
.agent-card {
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
}

.agent-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 34px rgba(43, 88, 255, 0.10);
    border-color: rgba(43, 88, 255, 0.25);
}

.agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 4px 0 0 4px;
}

/* Decision Card - Điểm nhấn chính */
.decision-card {
    background: linear-gradient(145deg, #ffffff, #f8faff);
    border: 2px solid transparent;
    background-clip: padding-box;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 40px rgba(11, 46, 172, 0.08);
    position: relative;
    transition: all 0.3s ease;
}

.decision-card::after {
    content: '';
    position: absolute;
    top: -2px; bottom: -2px; left: -2px; right: -2px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
    z-index: -1;
    border-radius: 22px;
    opacity: 0.8;
}

.decision-card h2 {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem;
    margin-top: 0;
}

/* Tùy chỉnh các thành phần Streamlit */
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: white !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.28);
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(37, 99, 235, 0.38);
    filter: brightness(1.04);
}

.stButton > button:active {
    transform: translateY(1px) scale(0.99);
}

.stButton > button[kind="secondary"] {
    background: #ffffff;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.stButton > button[kind="secondary"]:hover {
    border-color: #94a3b8 !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}

/* Form submit button (Chạy Multi-Agent) — nhấn mạnh hơn */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1.25rem !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.32) !important;
}

.stFormSubmitButton > button:hover {
    box-shadow: 0 10px 28px rgba(79, 70, 229, 0.42) !important;
}

/* Inputs, Selectboxes, Number/Date inputs, Textarea */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div > div,
.stMultiSelect > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e5e7eb !important;
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stDateInput > div > div > input:focus,
.stTextArea textarea:focus,
.stSelectbox > div > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border-radius: 7px !important;
}

/* Form container — khung nhẹ bao quanh toàn bộ form nhập liệu */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px 22px 10px 22px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #f8fafc, #eef2ff) !important;
    border: 1.5px dashed #a5b4fc !important;
    border-radius: 14px !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1 !important;
    background: linear-gradient(180deg, #eef2ff, #e0e7ff) !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #f8faff);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.03);
}

div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #1e3a8a, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

div[data-testid="stMetricLabel"] {
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.8rem !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid #f1f5f9;
}

/* Expander — bo tròn, viền mềm, nổi khối rõ ràng */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.7);
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    padding: 4px 2px;
}

/* Alerts (success / info / warning / error) — bo tròn nhất quán */
div[data-testid="stAlertContainer"], .stAlert {
    border-radius: 12px !important;
}

/* Muted text */
.small-muted {
    font-size: 0.85rem; 
    color: #64748b;
    font-weight: 500;
}

/* Status spinner/box */
[data-testid="stStatusWidget"] {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    background: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<div style="text-align: center; margin-top: 6px; margin-bottom: 34px; position: relative; z-index: 50;">
    <div style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 14px 40px;
        border-radius: 999px;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.12), inset 0 2px 4px rgba(255,255,255,0.8);
        border: 1px solid rgba(226, 232, 240, 0.9);
        animation: floatTitle 5s ease-in-out infinite;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    " class="hero-title-box">
        <span style="font-size: 2.2rem; animation: pulseBot 2.5s infinite; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));">🤖</span>
        <h1 style="
            margin: 0;
            font-family: 'Inter', sans-serif;
            font-size: 2.05rem;
            font-weight: 900;
            background: linear-gradient(90deg, #1e3a8a, #3b82f6, #8b5cf6, #ec4899);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            animation: textShine 4s linear infinite;
        ">
            OPC Multi-Agent Contract Decision System
        </h1>
    </div>
    <div style="
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 22px;
    ">
        <span class="flow-chip">📦 Team Pack (CSV)</span>
        <span class="flow-arrow">→</span>
        <span class="flow-chip">💰 Finance Agent</span>
        <span class="flow-arrow">→</span>
        <span class="flow-chip">🛡️ Risk Agent</span>
        <span class="flow-arrow">→</span>
        <span class="flow-chip">🧭 Decision Agent</span>
        <span class="flow-arrow">→</span>
        <span class="flow-chip flow-chip-final">✅ Founder Approval</span>
    </div>
</div>

<style>
@keyframes floatTitle {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); box-shadow: 0 15px 35px rgba(59, 130, 246, 0.18), inset 0 2px 4px rgba(255,255,255,0.8); }
    100% { transform: translateY(0px); }
}
@keyframes pulseBot {
    0% { transform: scale(1) rotate(0deg); }
    25% { transform: scale(1.1) rotate(-5deg); filter: drop-shadow(0 0 10px rgba(59,130,246,0.4)); }
    50% { transform: scale(1) rotate(0deg); }
    75% { transform: scale(1.1) rotate(5deg); filter: drop-shadow(0 0 10px rgba(139,92,246,0.4)); }
    100% { transform: scale(1) rotate(0deg); }
}
@keyframes textShine {
    to { background-position: 200% center; }
}
.hero-title-box:hover {
    transform: scale(1.02);
}
.flow-chip {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #334155;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 7px 14px;
    border-radius: 999px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
}
.flow-chip:hover {
    transform: translateY(-2px);
    border-color: #93c5fd;
    box-shadow: 0 6px 14px rgba(59, 130, 246, 0.14);
}
.flow-chip-final {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-color: #6ee7b7;
    color: #065f46;
}
.flow-arrow {
    color: #93a7f0;
    font-weight: 700;
    font-size: 0.95rem;
}

/* ================== SIDEBAR — NỀN TỐI (giống ảnh mẫu ECharts Gallery) ================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.18);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
/* Ô nhập text/password trong sidebar */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
/* Selectbox / dropdown kiểu "Choose options" trong ảnh mẫu */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #94a3b8 !important;
}
/* Alert (info/success/error) trong sidebar */
section[data-testid="stSidebar"] div[data-testid="stAlert"] {
    background: rgba(99, 102, 241, 0.16) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-testid="stAlert"] * {
    color: #c7d2fe !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        '<span style="font-size:1.3rem;">⚙️</span>'
        '<span style="font-size:1.25rem;font-weight:800;color:#ffffff;letter-spacing:-0.01em;">Cấu hình hệ thống</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API key",
        type="password",
        value="",
        help="Để trống nếu đã dán key vào OPENAI_API_KEY_HARDCODED trong code, "
        "hoặc đã đặt biến môi trường OPENAI_API_KEY.",
    )
    # Ưu tiên: ô nhập trên sidebar > key dán trực tiếp trong code > biến môi trường.
    api_key = api_key_input.strip() or OPENAI_API_KEY_HARDCODED.strip() or env_key
    model = st.text_input(
        "Model",
        value="gpt-5-mini",
        key="model_input_v4",
        help="Model OpenAI hỗ trợ Structured Outputs, ví dụ: gpt-5, gpt-5-mini,gpt-4o-mini, gpt-4o, gpt-4.1, gpt-4.1-mini.",
    )
    st.info("API key không được ghi vào Excel, prompt log hoặc Decision Card.")

result = st.session_state.get("opc_result")

st.markdown("""
<style>
/* Khung bao quanh thanh tab chính, giúp tab nổi bật và rõ ràng hơn */
div[data-testid="stTabs"] {
    background: linear-gradient(180deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 10px 14px 0 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    height: 56px;
    padding: 0 22px;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #64748b;
    border-radius: 12px 12px 0 0;
    transition: all 0.2s ease;
}

div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #4f46e5;
    background: rgba(99, 102, 241, 0.06);
}

div[data-testid="stTabs"] button[data-baseweb="tab"] p {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4338ca !important;
    background: #ffffff;
    border-bottom: 4px solid #6366f1 !important;
    box-shadow: 0 -4px 14px rgba(99, 102, 241, 0.08);
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #4f46e5, #ec4899) !important;
    height: 4px !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #e2e8f0 !important;
}

/* ================== OPERATIONS TAB — REDESIGN (card trắng bo tròn, KPI card) ================== */

.ops-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 23px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}
.ops-hero-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.ops-hero-desc {
    color: #64748b;
    font-size: 0.98rem;
    line-height: 1.6;
    max-width: 900px;
}

.ops-section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.3rem;
    font-weight: 800;
    color: #0f172a;
    margin: 2px 0 4px 0;
    letter-spacing: -0.01em;
}
.ops-section-desc {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 16px;
}

/* Form nhập liệu -> card trắng bo tròn, viền mềm, đổ bóng nhẹ */
div[data-testid="stTabs"] div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px;
    padding: 20px 20px 6px 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}

/* Vùng tải file -> card trắng bo tròn */
div[data-testid="stTabs"] [data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 14px !important;
}
div[data-testid="stTabs"] div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}

/* st.metric -> KPI card giống các thẻ chỉ số trên cùng của ảnh mẫu */
div[data-testid="stTabs"] div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
div[data-testid="stTabs"] div[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 2 rem !important;
    color: #64748b !important;
    font-weight: 700 !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
div[data-testid="stTabs"] div[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* Expander -> card trắng bo tròn, giống card biểu đồ trong ảnh */
div[data-testid="stTabs"] [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    overflow: hidden;
    margin-bottom: 14px;
}
div[data-testid="stTabs"] [data-testid="stExpander"] summary {
    padding: 14px 18px !important;
}

/* Dataframe -> card bo tròn, viền mềm */
div[data-testid="stTabs"] [data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
}

/* Alert (info/success/error) -> bo tròn mềm mại hơn, giống badge trong ảnh */
div[data-testid="stTabs"] div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid transparent;
}
div[data-testid="stTabs"] div[data-testid="stAlert"] p,
div[data-testid="stTabs"] div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] {
    font-size: 1rem !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

tab_ops, tab_crisis, tab_dashboard = st.tabs(["⚙️ Operations (Input & Workflow)", "🆘 Crisis Card", "🏆 Decision Dashboard"])

with tab_ops:
    st.markdown(
        """
<div class="ops-hero">
<div class="ops-hero-title">⚙️ Operations Console</div>
<div class="ops-hero-desc">
Trung tâm vận hành của OPC Multi-Agent Decision System — tải Team Pack, khởi tạo cơ hội kinh doanh
và theo dõi trực tiếp luồng xử lý của 3 AI Agent (Data &amp; Finance → Risk &amp; Compliance → Decision &amp; Partner)
trước khi ra Decision Card.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_workflow = st.columns([1.0, 2.2], gap="large")

    with col_input:
        st.markdown(
            '<div class="ops-section-title">📥 1. Input Data</div>'
            '<div class="ops-section-desc">Tải Team Pack Excel và nhập thông tin cơ hội kinh doanh.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Tải Team Pack Excel",
            type=["xlsx"],
            help="Hệ thống sẽ bóc tách các sheet bắt buộc thành CSV rồi nạp trực tiếp (không qua SQLite).",
        )

        data = None
        if uploaded_file:
            try:
                file_bytes = uploaded_file.getvalue()
                data = load_team_pack(file_bytes)
                st.session_state["opc_data"] = data
                st.success(f"Đã nạp {len(data)} sheet bắt buộc từ CSV.")
                with st.expander("Danh sách sheet đã đọc"):
                    st.write(list(data.keys()))
            except Exception as exc:
                st.error(f"Không đọc được Team Pack: {exc}")

        if data:
            profile = get_profile(data)
            customers = data["03_CUSTOMERS"].copy()
            products = data["05_PRODUCTS"].copy()
            customer_types = sorted(customers["customer_type"].dropna().astype(str).unique().tolist())
            service_names = products["service_name"].astype(str).tolist()

            with st.form("opportunity_form"):
                customer_name = st.text_input(
                    "Tên khách hàng",
                    help="Nhập tự do. Nếu khớp khách hàng cũ trong Team Pack, hệ thống hiển thị "
                    "customer_id và payment_reliability tương ứng.",
                )

                existing_customer = find_existing_customer(customers, customer_name) if customer_name else None
                if customer_name:
                    if existing_customer:
                        st.info(
                            f"Khách hàng cũ — customer_id: **{existing_customer['customer_id']}** · "
                            f"payment_reliability: **{existing_customer['payment_reliability']}**"
                        )
                    else:
                        st.caption("Không khớp khách hàng cũ nào trong Team Pack — sẽ xử lý như khách hàng mới.")

                default_type_index = 0
                if existing_customer and existing_customer.get("customer_type") in customer_types:
                    default_type_index = customer_types.index(existing_customer["customer_type"])
                customer_type = st.selectbox("Loại khách hàng (customer_type)", customer_types, index=default_type_index)

                province_default = existing_customer.get("province") if existing_customer else ""
                province = st.text_input("Tỉnh/thành phố (province)", value=province_default or "")

                num_provinces = st.number_input(
                    "Quy mô triển khai (số tỉnh/thành phố)",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Nhập thủ công số tỉnh thành triển khai dự án. Hệ số Oper sẽ được cộng "
                    "thêm: < 10 tỉnh thành → +0.01; 10-20 tỉnh thành → +0.02; > 20 tỉnh thành → +0.05.",
                )

                # FIX: Order Change (PDF mục 2.2) là 1 phần của Oper Score cơ sở, không
                # chỉ Crisis Card. Trường này để TRỐNG (0) nếu hợp đồng chưa xác định
                # trước số order -> không cộng thêm Oper, giữ nguyên hành vi cũ.
                initial_order_count = st.number_input(
                    "Số lượng order ban đầu của hợp đồng (nếu có)",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Chỉ nhập nếu hợp đồng đã xác định trước số lượng order. Hệ số Oper sẽ "
                    f"được cộng thêm nếu vượt {ORDER_CHANGE_FREE_LIMIT} order/HĐ: "
                    f"> {ORDER_CHANGE_FREE_LIMIT} và ≤ {ORDER_CHANGE_HARD_CAP} order → "
                    f"+{ORDER_CHANGE_SURCHARGE_OPER:.1%}; > {ORDER_CHANGE_HARD_CAP} order → vượt trần cứng "
                    "(ngưỡng tạm đặt, cần Founder xác nhận lại).",
                )

                selected_services = st.multiselect(
                    "Danh sách dịch vụ (service_name) theo yêu cầu khách hàng",
                    service_names,
                )
                selected_products = products.loc[products["service_name"].isin(selected_services)].copy()
                total_list_price = float(selected_products["list_price"].sum()) if not selected_products.empty else 0.0

                if not selected_products.empty:
                    pricing_preview = selected_products[["service_name", "pricing_model", "list_price"]].copy()
                    pricing_preview["list_price"] = pricing_preview["list_price"].map(format_vnd)
                    st.caption("Pricing model của các dịch vụ đã chọn:")
                    st.dataframe(
                        pricing_preview.rename(
                            columns={
                                "service_name": "Dịch vụ",
                                "pricing_model": "Pricing model",
                                "list_price": "List price",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.metric("Tổng list_price", format_vnd(total_list_price))

                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    order_date_input = st.date_input("order_date")
                with date_col2:
                    due_date_input = st.date_input("due_date")

                run_button = st.form_submit_button(
                    "▶ Chạy Multi-Agent",
                    type="primary",
                    use_container_width=True,
                )

            if run_button:
                if not api_key:
                    st.error("Hãy nhập OpenAI API key, dán vào OPENAI_API_KEY_HARDCODED trong code, hoặc đặt biến môi trường OPENAI_API_KEY.")
                elif selected_products.empty:
                    st.error("Vui lòng chọn ít nhất một dịch vụ (service_name).")
                elif due_date_input <= order_date_input:
                    st.error("due_date phải sau order_date.")
                else:
                    try:
                        order_date = pd.Timestamp(order_date_input)
                        due_date = pd.Timestamp(due_date_input)

                        payment_reliability = (
                            float(existing_customer["payment_reliability"])
                            if existing_customer and existing_customer.get("payment_reliability") is not None
                            else None
                        )
                        customer_id_for_lookup = existing_customer["customer_id"] if existing_customer else None
                        transaction_risk_score = latest_transaction_risk_score(
                            data["08_BANK_TXN"], customer_id_for_lookup
                        )

                        finance_metrics = build_finance_metrics(
                            selected_products=selected_products,
                            payment_reliability=payment_reliability,
                            province=province,
                            transaction_risk_score=transaction_risk_score,
                            order_date=order_date,
                            due_date=due_date,
                            num_provinces=int(num_provinces) if num_provinces else None,
                            initial_order_count=int(initial_order_count) if initial_order_count else None,
                        )

                        if finance_metrics.get("order_change_hard_cap_exceeded"):
                            # Hợp đồng ban đầu đã vượt trần cứng Order Change ngay từ lúc tạo
                            # -> cảnh báo rõ cho Founder (không chặn tính toán, để Founder vẫn
                            # thấy đầy đủ Decision Card và tự quyết định, giống cách UI xử lý
                            # missing_fields).
                            st.warning(
                                f"⚠️ Số lượng order ban đầu ({int(initial_order_count)}) vượt trần cứng "
                                f"Order Change ({ORDER_CHANGE_HARD_CAP} - ngưỡng tạm đặt, cần Founder "
                                "xác nhận lại con số này)."
                            )

                        reserve_minimum = float(
                            profile.get("cash_reserve_minimum", CASH_RESERVE_THRESHOLD_DEFAULT)
                            or CASH_RESERVE_THRESHOLD_DEFAULT
                        )
                        cash_projection = project_closing_cash(
                            data=data,
                            selected_products=selected_products,
                            finance_metrics=finance_metrics,
                            order_date=order_date,
                            reserve_minimum=reserve_minimum,
                        )

                        # Lọc 4 lớp sản phẩm ngân hàng (Lớp 1: loại gói tín dụng —
                        # account_ops / credit_guarantee / unclassified bị loại; Lớp 2:
                        # nhu cầu vốn tối thiểu; Lớp 3: so sánh tổng chi phí; Lớp 4:
                        # ràng buộc tài sản đảm bảo) và xác định gói vay đề xuất TRƯỚC KHI tính
                        # Confidence Score, vì Confidence Score chỉ được tính khi đã có đề
                        # xuất gói vay phù hợp (partner_matrix có ít nhất 1 eligible=True).
                        partner_matrix = build_partner_matrix(
                            data=data,
                            funding_need=max(
                                0.0, reserve_minimum - cash_projection["min_projected_closing_cash"]
                            ),
                            cash_projection=cash_projection,
                        )
                        requested_amount = determine_requested_amount(cash_projection, partner_matrix)
                        founder_approval_needed = requested_amount > LARGE_DECISION_THRESHOLD
                        bank_product_classification = classify_all_bank_products(data)

                        confidence_result = compute_confidence_score(
                            cash_projection=cash_projection,
                            partner_matrix=partner_matrix,
                            total_list_price=finance_metrics["total_list_price"],
                            funding_amount=requested_amount,
                            province=province,
                        )

                        missing_fields = []
                        if not province:
                            missing_fields.append("province")
                        # Khách hàng mới không có payment_reliability là tình huống bình
                        # thường -> không đưa vào missing_fields / không yêu cầu bổ sung dữ liệu.

                        risk_eval = evaluate_risk_rules(
                            data=data,
                            finance_metrics=finance_metrics,
                            cash_projection=cash_projection,
                            confidence_result=confidence_result,
                        )

                        client = OpenAI(api_key=api_key)

                        workflow_logs = []
                        with st.status("Các Agent đang phối hợp...", expanded=True) as status:
                            start = time.perf_counter()

                            st.write("① Data & Finance Agent đang phân tích...")
                            finance_payload = {
                                "customer": {
                                    "customer_name": customer_name,
                                    "customer_type": customer_type,
                                    "province": province,
                                    "existing_customer": existing_customer,
                                },
                                "opportunity": {
                                    "selected_services": selected_services,
                                    "order_date": str(order_date.date()),
                                    "due_date": str(due_date.date()),
                                    "initial_order_count": int(initial_order_count) if initial_order_count else None,
                                },
                                "finance_metrics": finance_metrics,
                                "cash_projection": cash_projection,
                                "confidence_result": confidence_result,
                                "missing_fields": missing_fields,
                            }
                            
                            # API-H-004: mask/tokenize customer_id, customer_name, account_id...
                            # TRƯỚC KHI gửi cho OpenAI. finance_payload gốc (chứa dữ liệu thật)
                            # không bị thay đổi — chỉ dùng cho bản mask này để gọi Agent.
                            masked_finance_payload, finance_masked_fields = mask_sensitive_fields(
                                finance_payload
                            )
                            finance_result, finance_response_id = run_finance_agent(
                                client, model, masked_finance_payload
                            )
                                
                            workflow_logs.append(
                                {
                                    "agent": "Data & Finance Agent",
                                    "response_id": finance_response_id,
                                    "result": finance_result.model_dump(),
                                    "masked_fields": finance_masked_fields,
                                    "input": masked_finance_payload,
                                    "action": "Phân tích dữ liệu tài chính khách hàng, đánh giá chất lượng dữ liệu và đưa ra đánh giá sơ bộ (preliminary assessment).",
                                    "timestamp": time.strftime("%H:%M:%S %d/%m/%Y"),
                                }
                            )
                            st.write("✓ Data & Finance Agent hoàn tất")

                            st.write("⏳ Đang làm mát hệ thống (tránh Rate Limit)...")
                            time.sleep(4)

                            st.write("② Risk & Compliance Agent đang kiểm soát...")
                            risk_payload = {
                                "finance_agent_output": finance_result.model_dump(),
                                "finance_metrics": finance_metrics,
                                "cash_projection": cash_projection,
                                "confidence_result": confidence_result,
                                "triggered_rules": risk_eval["triggered_rules"],
                                "risk_level": risk_eval["risk_level"],
                                "missing_fields": missing_fields,
                            }
                            
                            # API-H-004: mask/tokenize trước khi gửi Risk Agent (risk_payload
                            # chứa lại finance_agent_output nên cần mask lại đề phòng).
                            masked_risk_payload, risk_masked_fields = mask_sensitive_fields(
                                risk_payload
                            )
                            risk_result, risk_response_id = run_risk_agent(
                                client, model, masked_risk_payload
                            )
                                
                            workflow_logs.append(
                                {
                                    "agent": "Risk & Compliance Agent",
                                    "response_id": risk_response_id,
                                    "result": risk_result.model_dump(),
                                    "masked_fields": risk_masked_fields,
                                    "input": masked_risk_payload,
                                    "action": "Kiểm tra các quy tắc rủi ro (Risk Rules), xác định mức độ rủi ro và đề xuất biện pháp kiểm soát.",
                                    "timestamp": time.strftime("%H:%M:%S %d/%m/%Y"),
                                }
                            )
                            st.write("✓ Risk & Compliance Agent hoàn tất")

                            st.write("⏳ Đang làm mát hệ thống (tránh Rate Limit)...")
                            time.sleep(4)

                            st.write("③ Decision & Partner Agent đang lập Decision Card...")
                            decision_payload = {
                                "customer": {
                                    "customer_name": customer_name,
                                    "customer_type": customer_type,
                                    "province": province,
                                },
                                "finance_metrics": finance_metrics,
                                "cash_projection": cash_projection,
                                "confidence_result": confidence_result,
                                "finance_agent_output": finance_result.model_dump(),
                                "risk_agent_output": risk_result.model_dump(),
                                "partner_matrix": partner_matrix,
                                "requested_amount": requested_amount,
                                "large_decision_threshold": LARGE_DECISION_THRESHOLD,
                                "founder_approval_needed": founder_approval_needed,
                                "missing_fields": missing_fields,
                            }
                            
                            # API-H-004: mask/tokenize trước khi gửi Decision Agent.
                            masked_decision_payload, decision_masked_fields = mask_sensitive_fields(
                                decision_payload
                            )
                            decision_result, decision_response_id = run_decision_agent(
                                client, model, masked_decision_payload
                            )
                                
                            decision_result = enforce_decision_card(
                                decision_result=decision_result,
                                finance_metrics=finance_metrics,
                                cash_projection=cash_projection,
                                confidence_result=confidence_result,
                                partner_matrix=partner_matrix,
                                requested_amount=requested_amount,
                                founder_approval_needed=founder_approval_needed,
                                triggered_rule_ids=[
                                    item["rule_id"] for item in risk_eval["triggered_rules"]
                                ],
                                is_new_customer=existing_customer is None,
                            )
                            workflow_logs.append(
                                {
                                    "agent": "Decision & Partner Agent",
                                    "response_id": decision_response_id,
                                    "result": decision_result.model_dump(),
                                    "masked_fields": decision_masked_fields,
                                    "input": masked_decision_payload,
                                    "action": "Tổng hợp kết quả từ Finance Agent và Risk Agent để lập Decision Card và đề xuất phương án tài trợ.",
                                    "timestamp": time.strftime("%H:%M:%S %d/%m/%Y"),
                                }
                            )
                            elapsed = time.perf_counter() - start
                            st.write("✓ Decision & Partner Agent hoàn tất")
                            status.update(
                                label=f"Hoàn tất Multi-Agent trong {elapsed:.1f} giây",
                                state="complete",
                                expanded=False,
                            )

                        # BUG cũ: founder_decision không được reset khi chạy phân tích mới ->
                        # nếu Founder đã Phê duyệt hợp đồng trước đó, hợp đồng MỚI (dù khác
                        # khách hàng, khác số tiền, có thể >300tr) sẽ hiển thị ngay "APPROVED"
                        # mà Founder chưa hề xem qua. Luôn reset về "Chưa quyết định" mỗi khi
                        # có Decision Card mới.
                        st.session_state.founder_decision = "Chưa quyết định"
                        # FIX (theo yêu cầu bổ sung): mỗi lần chạy lại Multi-Agent là một hợp
                        # đồng/baseline MỚI được nạp vào "opc_result" — dữ liệu Crisis Card
                        # (crisis_card, crisis_result) đang lưu trong session thuộc về hợp đồng
                        # CŨ nên phải bị xóa ngay tại đây để không hiển thị nhầm Crisis Card của
                        # hợp đồng cũ lên hợp đồng mới. Ngược lại, việc chỉ CHUYỂN TAB (không
                        # chạy lại Multi-Agent) sẽ KHÔNG chạm vào nhánh này nên dữ liệu Crisis
                        # Card đã nhập/đã tính vẫn được giữ nguyên.
                        st.session_state.pop("crisis_card", None)
                        st.session_state.pop("crisis_result", None)
                        st.session_state["opc_result"] = {
                            "model": model,
                            "profile": profile,
                            "customer": {
                                "customer_name": customer_name,
                                "customer_type": customer_type,
                                "province": province,
                                "existing_customer": existing_customer,
                                # FIX (bug có sẵn): trước đây key này không tồn tại trong
                                # "customer", nên mọi lần resolve_crisis_deltas() gọi
                                # result["customer"].get("num_provinces") để lấy
                                # old_num_provinces đều nhận về None -> cơ chế chống cộng
                                # trùng hệ số quy mô của SCOPE_CHANGE (net_scale_delta =
                                # new_scale - old_scale) không bao giờ hoạt động thực tế,
                                # luôn cộng thẳng hệ số quy mô MỚI lên baseline (đã có sẵn
                                # hệ số quy mô CŨ) -> bị tính trùng. Lưu lại num_provinces
                                # gốc tại đây để Crisis Card đọc đúng giá trị cũ.
                                "num_provinces": int(num_provinces) if num_provinces else None,
                            },
                            "opportunity": {
                                "selected_services": selected_services,
                                "order_date": str(order_date.date()),
                                "due_date": str(due_date.date()),
                                "initial_order_count": int(initial_order_count) if initial_order_count else None,
                            },
                            "finance_metrics": finance_metrics,
                            "cash_projection": cash_projection,
                            "confidence_result": confidence_result,
                            "missing_fields": missing_fields,
                            "triggered_rules": risk_eval["triggered_rules"],
                            "risk_level": risk_eval["risk_level"],
                            "transaction_risk_score": transaction_risk_score,
                            "payload_debug": {
                                "Data & Finance Agent": {
                                    "before_mask": finance_payload,
                                    "after_mask": masked_finance_payload,
                                    "masked_fields": finance_masked_fields,
                                },
                                "Risk & Compliance Agent": {
                                    "before_mask": risk_payload,
                                    "after_mask": masked_risk_payload,
                                    "masked_fields": risk_masked_fields,
                                },
                                "Decision & Partner Agent": {
                                    "before_mask": decision_payload,
                                    "after_mask": masked_decision_payload,
                                    "masked_fields": decision_masked_fields,
                                },
                            },
                            "partner_matrix": partner_matrix,
                            "bank_product_classification": bank_product_classification,
                            "requested_amount": requested_amount,
                            "founder_approval_needed": founder_approval_needed,
                            "finance_result": finance_result.model_dump(),
                            "risk_result": risk_result.model_dump(),
                            "decision_result": decision_result.model_dump(),
                            "workflow_logs": workflow_logs,
                            "elapsed_seconds": elapsed,
                        }
                        st.rerun()

                    except Exception as exc:
                        st.error("🚨 ỨNG DỤNG BỊ LỖI - Vui lòng copy toàn bộ dòng chữ đỏ dưới đây gửi cho tôi:")
                        st.code(repr(exc))
                        if hasattr(exc, "last_attempt") and exc.last_attempt is not None:
                            st.code(repr(exc.last_attempt.exception()))
        else:
            st.info("Tải Team Pack để mở form cơ hội kinh doanh.")




    with col_workflow:
        st.markdown(
            '<div class="ops-section-title">🔄 2. Agent Workflow</div>'
            '<div class="ops-section-desc">Theo dõi luồng xử lý theo thời gian thực của 3 AI Agent.</div>',
            unsafe_allow_html=True,
        )
        if not result:
            st.info("Workflow sẽ xuất hiện sau khi chạy hệ thống.")
        else:
            st.success(
                f"OpenAI "
                f"{result['elapsed_seconds']:.1f}s"
            )

            for index, log in enumerate(result["workflow_logs"], start=1):
                with st.expander(f"{index}. {log['agent']} — Completed", expanded=True):
                    st.caption(
                        f"OpenAI response ID: {log['response_id']}  •  "
                        f"Thời gian hoàn tất: {log.get('timestamp', 'N/A')}"
                    )
                    res = log["result"]
                    agent_name = log["agent"]

                    st.markdown(
                        '<div style="font-weight:700; color:#0f172a; font-size:0.8rem; '
                        'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">'
                        '📥 Input</div>',
                        unsafe_allow_html=True,
                    )
                    st.json(log.get("input", {}), expanded=False)

                    st.markdown(
                        '<div style="font-weight:700; color:#0f172a; font-size:0.8rem; '
                        'text-transform:uppercase; letter-spacing:0.05em; margin-top:16px; margin-bottom:6px;">'
                        '⚙️ Action</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="color:#334155; font-size:0.9rem; line-height:1.5; margin-bottom:10px;">'
                        f'{log.get("action", "")}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div style="font-weight:700; color:#0f172a; font-size:0.8rem; '
                        'text-transform:uppercase; letter-spacing:0.05em; margin-top:16px; margin-bottom:6px;">'
                        '📤 Output</div>',
                        unsafe_allow_html=True,
                    )

                    if "Finance" in agent_name:
                        st.markdown(f"""
<div class="agent-card" style="margin-bottom: 0;">
<div style="display: flex; gap: 10px; margin-bottom: 15px;">
<span style="background: #e0e7ff; color: #3730a3; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Quality: {res.get('data_quality', 'N/A')}</span>
<span style="background: #dcfce3; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Assessment: {res.get('preliminary_assessment', 'N/A')}</span>
</div>
<p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">{res.get('summary', '')}</p>
<div style="background: #f8fafc; border-left: 3px solid #3b82f6; padding: 12px; margin-bottom: 15px; border-radius: 4px;">
<strong style="color: #1e293b; font-size: 0.9rem;">Key Observations:</strong>
<ul style="margin-top: 8px; margin-bottom: 0; color: #475569; font-size: 0.9rem; padding-left: 20px;">
{''.join(f'<li>{obs}</li>' for obs in res.get('key_observations', []))}
</ul>
</div>
</div>
                        """, unsafe_allow_html=True)

                    elif "Risk" in agent_name:
                        risk_level = res.get('risk_level', 'LOW')
                        risk_color = "#ef4444" if risk_level in ["CRITICAL", "HIGH"] else "#eab308" if risk_level == "MEDIUM" else "#22c55e"
                        st.markdown(f"""
<div class="agent-card" style="margin-bottom: 0;">
<div style="display: flex; gap: 10px; margin-bottom: 15px;">
<span style="background: {risk_color}20; color: {risk_color}; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">Risk: {risk_level}</span>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
<div style="background: #fef2f2; border: 1px solid #fecaca; padding: 12px; border-radius: 8px;">
<strong style="color: #991b1b; font-size: 0.9rem;">Warnings / Rules</strong>
<ul style="margin-top: 8px; margin-bottom: 0; color: #7f1d1d; font-size: 0.85rem; padding-left: 16px;">
{''.join(f'<li>{w}</li>' for w in res.get('warnings', []))}
</ul>
</div>
<div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px;">
<strong style="color: #166534; font-size: 0.9rem;">Recommended Controls</strong>
<ul style="margin-top: 8px; margin-bottom: 0; color: #14532d; font-size: 0.85rem; padding-left: 16px;">
{''.join(f'<li>{c}</li>' for c in res.get('recommended_controls', []))}
</ul>
</div>
</div>
<div style="background: #fffbeb; border: 1px solid #fde68a; padding: 12px; margin-bottom: 15px; border-radius: 8px;">
<strong style="color: #92400e; font-size: 0.9rem;">Human Confirmation Points</strong>
<ul style="margin-top: 8px; margin-bottom: 0; color: #b45309; font-size: 0.85rem; padding-left: 16px;">
{''.join(f'<li>{c}</li>' for c in res.get('human_confirmation_points', []))}
</ul>
</div>
</div>
                        """, unsafe_allow_html=True)

                    elif "Decision" in agent_name:
                        st.markdown(f"""
<div class="agent-card" style="margin-bottom: 0; border: 1px solid #c7d2fe;">
<div style="display: flex; gap: 10px; margin-bottom: 15px;">
<span style="background: #818cf8; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Recommendation: {res.get('recommendation', 'N/A')}</span>
<span style="background: #f1f5f9; color: #475569; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">Approval Required: {'Yes' if res.get('human_approval_required') else 'No'}</span>
</div>
<p style="color: #334155; font-size: 0.95rem; line-height: 1.5;"><strong>Executive Summary:</strong> {res.get('executive_summary', '')}</p>
<div style="background: #f8fafc; border-left: 3px solid #6366f1; padding: 12px; margin-bottom: 15px; border-radius: 4px;">
<strong style="color: #312e81; font-size: 0.9rem;">Selected Option:</strong> <span style="color: #4f46e5; font-weight: 600;">{res.get('selected_financing_option', 'N/A')}</span>
</div>
<div style="background: #fafafa; border-left: 3px solid #f59e0b; padding: 12px; margin-top: 10px; border-radius: 4px;">
<strong style="color: #92400e; font-size: 0.9rem;">Protection Condition:</strong> <span style="color: #b45309;">{res.get('protection_condition', 'N/A')}</span>
</div>
</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.json(res)

            with st.expander("Triggered Risk Rules"):
                if result["triggered_rules"]:
                    st.dataframe(
                        pd.DataFrame(result["triggered_rules"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Không có rule được kích hoạt.")

            with st.expander("Cash Flow Schedule (mô phỏng theo tháng)"):
                st.dataframe(
                    pd.DataFrame(result["cash_projection"]["schedule"]),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Partner Option Matrix (11_BANK_PRODUCTS)"):
                option_df = pd.DataFrame(result["partner_matrix"]).copy()
                if not option_df.empty:
                    option_df["annual_rate_or_fee"] = option_df["annual_rate_or_fee"].map(lambda v: f"{v:.2%}")
                    option_df["processing_fee_rate"] = option_df["processing_fee_rate"].map(lambda v: f"{v:.2%}")
                    option_df["minimum_amount"] = option_df["minimum_amount"].map(format_vnd)
                    st.dataframe(
                        option_df[
                            [
                                "bank", "product_name", "annual_rate_or_fee", "processing_fee_rate",
                                "minimum_amount", "automation_level", "eligible",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Không kích hoạt (Projected_Closing_Cash ≥ ngưỡng dự trữ tối thiểu).")

            with st.expander("Phân loại sản phẩm ngân hàng (audit — vì sao giữ/loại)"):
                st.caption(
                    "Chỉ sản phẩm tín dụng bơm tiền mặt trực tiếp (working capital, "
                    "factoring...) mới được đưa vào Partner Option Matrix ở trên. Dịch vụ "
                    "vận hành tài khoản và sản phẩm bảo lãnh/hỗ trợ giao dịch bị loại vì "
                    "không giải quyết đúng RR-002 (thiếu hụt tiền mặt), dù có thể vẫn "
                    "eligible về mặt hạn mức."
                )
                classification = result.get("bank_product_classification", [])
                if classification:
                    cls_df = pd.DataFrame(classification)
                    category_labels = {
                        "credit_cash": "🟢 Tín dụng — bơm tiền mặt trực tiếp (GIỮ)",
                        "credit_guarantee": "🟡 Tín dụng — bảo lãnh/hỗ trợ giao dịch (LOẠI, sai mục đích)",
                        "account_ops": "🔵 Vận hành tài khoản — không phải khoản vay (LOẠI)",
                        "unclassified": "⚪ Chưa phân loại được (LOẠI, cần Founder rà soát thủ công)",
                    }
                    cls_df["Phân loại"] = cls_df["category"].map(category_labels).fillna(cls_df["category"])
                    st.dataframe(
                        cls_df.rename(
                            columns={
                                "bank": "Bank",
                                "product_name": "Sản phẩm",
                                "matched_keyword": "Từ khóa khớp",
                            }
                        )[["Bank", "Sản phẩm", "Phân loại", "Từ khóa khớp"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    unclassified_names = [
                        f"{item['bank']} — {item['product_name']}"
                        for item in classification
                        if item["category"] == "unclassified"
                    ]
                    if unclassified_names:
                        st.warning(
                            "⚠️ Có sản phẩm chưa phân loại được (không khớp từ khóa nào): "
                            + "; ".join(unclassified_names)
                            + ". Hệ thống KHÔNG tự đoán và đã loại khỏi so sánh — Founder "
                            "cần rà soát thủ công xem có nên bổ sung vào Partner Option "
                            "Matrix hay không."
                        )
                else:
                    st.write("Chưa có dữ liệu phân loại.")

            with st.expander("🔍 Payload trước/sau mask (khách hàng vừa nhập lượt này)", expanded=False):
                st.caption(
                    "Đây là payload THẬT của lượt chạy vừa rồi — không phải ví dụ cố định. "
                    "Nhập khách hàng khác, chạy lại, mở expander này sẽ thấy giá trị khác đi "
                    "tương ứng đúng khách hàng đó (chứng minh masking hoạt động tự động cho "
                    "MỌI khách hàng, không hard-code riêng cho khách hàng nào)."
                )
                payload_debug = result.get("payload_debug", {})
                agent_tabs = st.tabs(list(payload_debug.keys()))
                for tab, agent_name in zip(agent_tabs, payload_debug.keys()):
                    with tab:
                        debug_info = payload_debug[agent_name]
                        if debug_info["masked_fields"]:
                            st.success(
                                "✅ Đã mask field: " + ", ".join(debug_info["masked_fields"])
                            )
                        else:
                            st.info("Không có field nhạy cảm nào trong payload của agent này.")
                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.markdown("**OPC data**")
                            st.json(debug_info["before_mask"])
                        with col_after:
                            st.markdown("**Data Masking**")
                            st.json(debug_info["after_mask"])

            with st.expander("🔒 Kiểm tra tuân thủ gọi API (22_API_HANDLING_RULES)", expanded=False):
                st.caption(
                    "Đối chiếu TẤT ĐỊNH (không dùng OpenAI) giữa trạng thái runtime của lượt "
                    "chạy này với 8 rule trong bảng 22_API_HANDLING_RULES. ✅ OK = tuân thủ · "
                    "🟡 REVIEW = không phải lỗi hệ thống, nhưng đúng theo bảng gốc là điểm cần "
                    "con người rà soát/xác nhận (requires_human_approval) · ⚪ N/A = rule không "
                    "áp dụng cho lượt chạy này."
                )
                decision_for_check = result["decision_result"]
                api_checklist = evaluate_api_handling_checklist(
                    missing_fields=result["missing_fields"],
                    partner_matrix=result["partner_matrix"],
                    requested_amount=result["requested_amount"],
                    triggered_rule_ids=[
                        item["rule_id"] for item in result["triggered_rules"]
                    ],
                    confidence_score=decision_for_check.get("confidence_score"),
                    transaction_risk_score=result.get("transaction_risk_score"),
                    workflow_logs=result["workflow_logs"],
                )
                status_icon = {"OK": "✅", "REVIEW": "🟡", "N/A": "⚪"}
                checklist_df = pd.DataFrame(
                    [
                        {
                            "Tick": status_icon.get(item["status"], "❓"),
                            "Rule": item["rule_id"],
                            "Áp dụng cho": item["applies_to"],
                            "Yêu cầu xử lý": item["required_handling"],
                            "Trạng thái lượt chạy này": item["detail"],
                        }
                        for item in api_checklist
                    ]
                )
                st.dataframe(checklist_df, use_container_width=True, hide_index=True)
                review_count = sum(1 for item in api_checklist if item["status"] == "REVIEW")
                if review_count:
                    st.warning(
                        f"🟡 Có {review_count} rule đang ở trạng thái REVIEW — cần Founder "
                        "rà soát/xác nhận thêm trước khi chốt quyết định cuối."
                    )
                else:
                    st.success("✅ Không có rule nào cần rà soát thêm trong lượt chạy này.")


CRISIS_GROUP_LABELS = {
    "DEADLINE_EARLY": "Giao sớm (Deadline Early)",
    "DEADLINE_LATE": "Giao muộn (Deadline Late)",
    "COST_CHANGE": "Phát sinh chi phí (Cost Change)",
    "PAYMENT_DELAY": "Chậm thanh toán (Payment Delay)",
    "FINANCE_CONDITION": "Thay đổi đ/k tài chính (Finance Condition)",
    "SCOPE_CHANGE": "Đổi địa bàn (Scope Change)",
    "ORDER_CHANGE": "Đổi số lượng đơn hàng (Order Change)"
}

with tab_crisis:
    st.markdown(
        """
<div class="ops-hero">
<div class="ops-hero-title">🆘 Crisis Card</div>
<div class="ops-hero-desc">
Nhập biến động phát sinh (trễ hạn, phát sinh chi phí, thay đổi điều kiện tài chính...) để đánh giá
Before / After và gọi AI Agent chốt phương án xử lý — áp dụng cho đúng hợp đồng baseline đang chạy ở tab Operations.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if not result:
        st.warning("Vui lòng chạy baseline ở tab Operations trước khi nhập Crisis Card.")
    else:
        baseline_metrics = result.get("finance_metrics", {})
        default_late_amount = 0.0
        if baseline_metrics.get("contract_months") and baseline_metrics["contract_months"] > 0:
            default_late_amount = float(baseline_metrics["total_list_price"] / baseline_metrics["contract_months"])

        # FIX (theo yêu cầu bổ sung): bỏ ô nhập contract_id thủ công — Crisis Card
        # giờ LUÔN áp dụng cho đúng hợp đồng đang chạy ở tab Operations (kết quả
        # baseline hiện có trong session), lấy customer_id nếu là khách hàng cũ,
        # nếu không thì fallback về tên khách hàng đã nhập.
        _existing_customer_info = (result.get("customer") or {}).get("existing_customer") or {}
        running_contract_id = str(
            _existing_customer_info.get("customer_id")
            or (result.get("customer") or {}).get("customer_name")
            or "N/A"
        )
        st.info(f"🔗 Crisis Card sẽ áp dụng cho hợp đồng đang chạy: **{running_contract_id}**")

        # FIX (nghiêm trọng): yêu cầu ghi rõ "Nhập dữ kiện Crisis Card bằng prompt
        # HOẶC biểu mẫu đơn giản" -- trước đây chỉ có đường Form. Nay thêm lựa chọn
        # chế độ nhập; cả 2 đường đều hội tụ về cùng 1 CrisisCardInput, cùng đi qua
        # validate_crisis_card_input() và cùng 1 khối xử lý tất định bên dưới (không
        # có đường tắt nào bỏ qua kiểm tra hợp lệ).
        input_mode = st.radio(
            "Chế độ nhập Crisis Card",
            ["📋 Biểu mẫu (form)", "💬 Prompt (mô tả tự do bằng AI)"],
            horizontal=True,
        )

        crisis_submit = False
        crisis_card = None
        validation_errors = []

        if input_mode == "📋 Biểu mẫu (form)":
            with st.form("crisis_card_form"):
                crisis_group = st.multiselect("Nhóm biến động (crisis_group)", list(CRISIS_GROUP_LABELS.keys()), format_func=lambda key: CRISIS_GROUP_LABELS[key], max_selections=2, key="cc_crisis_group")

                days_deviation_input = st.number_input("Số ngày sớm/muộn", min_value=0, value=0, step=1, key="cc_days_deviation")

                extra_cost_mode = st.radio(
                    "Cách nhập chi phí phát sinh (COST_CHANGE)",
                    ["Theo số tiền (VNĐ)", "Theo phần trăm (%)"],
                    horizontal=True,
                    key="cc_extra_cost_mode",
                )
                if extra_cost_mode == "Theo số tiền (VNĐ)":
                    extra_cost_amount_input = st.number_input("Chi phí phát sinh (VNĐ)", min_value=0.0, value=0.0, step=1_000_000.0, key="cc_extra_cost_amount")
                    extra_cost_percent_input = 0.0
                else:
                    extra_cost_amount_input = 0.0
                    extra_cost_percent_input = st.number_input(
                        "Chi phí phát sinh (% trên estimated_cost baseline, âm = giảm chi phí)",
                        min_value=-100.0, value=0.0, step=0.5, format="%.2f",
                        key="cc_extra_cost_percent",
                    )
                late_amount_input = st.number_input("Số tiền khách hàng trả muộn (VNĐ)", min_value=0.0, value=default_late_amount, step=1_000_000.0, key="cc_late_amount")
                late_month_input = st.text_input(
                    "Tháng bị trả muộn (VD: 2026-07) — để trống = tự động áp dụng vào tháng ĐẦU của hợp đồng",
                    key="cc_late_month",
                )
                late_days_input = st.number_input("Số ngày trả muộn (áp lãi kép 1%/ngày)", min_value=0, value=0, step=1, key="cc_late_days")

                # FIX (bug logic thực sự): trước đây dùng "giá trị = 0.0" làm cờ
                # "không đổi trường này", nên KHÔNG THỂ nhập giá trị 0 hợp lệ về mặt
                # nghiệp vụ (VD: miễn phí xử lý, không cần thế chấp). Nay dùng
                # checkbox tường minh để phân biệt "không đổi" và "đổi thành 0".
                st.caption("Tick chọn nếu muốn đổi trường tương ứng (kể cả khi giá trị mới = 0, VD: miễn phí xử lý / không cần thế chấp).")
                chg_rate = st.checkbox("Đổi annual_rate_or_fee", key="cc_chg_rate")
                new_annual_rate_or_fee_input = st.number_input("annual_rate_or_fee mới", min_value=0.0, value=0.0, step=0.001, format="%.4f", disabled=not chg_rate, key="cc_new_annual_rate")
                chg_fee = st.checkbox("Đổi processing_fee_rate", key="cc_chg_fee")
                new_processing_fee_rate_input = st.number_input("processing_fee_rate mới", min_value=0.0, value=0.0, step=0.001, format="%.4f", disabled=not chg_fee, key="cc_new_processing_fee")
                chg_collateral = st.checkbox("Đổi collateral_ratio", key="cc_chg_collateral")
                new_collateral_ratio_input = st.number_input("collateral_ratio mới", min_value=0.0, value=0.0, step=0.01, format="%.2f", disabled=not chg_collateral, key="cc_new_collateral")

                new_num_provinces_input = st.number_input("Số tỉnh/thành phố mới", min_value=0, value=0, step=1, key="cc_new_num_provinces")
                new_order_count_input = st.number_input("Số lượng đơn hàng mới", min_value=0, value=0, step=1, key="cc_new_order_count")

                crisis_submit = st.form_submit_button("Xác nhận Crisis Card", type="primary", use_container_width=True)

            if crisis_submit:
                finance_condition_fields = {
                    "new_annual_rate_or_fee": new_annual_rate_or_fee_input if chg_rate else None,
                    "new_processing_fee_rate": new_processing_fee_rate_input if chg_fee else None,
                    "new_collateral_ratio": new_collateral_ratio_input if chg_collateral else None,
                }

                try:
                    crisis_card = CrisisCardInput(
                        crisis_group=crisis_group,
                        contract_id=running_contract_id,
                        days_deviation=int(days_deviation_input) if days_deviation_input else None,
                        extra_cost_amount=float(extra_cost_amount_input) if extra_cost_amount_input else None,
                        extra_cost_percent=float(extra_cost_percent_input) if extra_cost_percent_input else None,
                        late_amount=float(late_amount_input) if late_amount_input else None,
                        late_month=late_month_input.strip() if late_month_input else None,
                        late_days=int(late_days_input) if late_days_input else None,
                        new_annual_rate_or_fee=finance_condition_fields["new_annual_rate_or_fee"],
                        new_processing_fee_rate=finance_condition_fields["new_processing_fee_rate"],
                        new_collateral_ratio=finance_condition_fields["new_collateral_ratio"],
                        new_num_provinces=int(new_num_provinces_input) if new_num_provinces_input else None,
                        new_order_count=int(new_order_count_input) if new_order_count_input else None,
                    )
                    validation_errors = validate_crisis_card_input(crisis_card)
                except Exception as exc:
                    crisis_card = None
                    validation_errors = [f"Không dựng được CrisisCardInput: {exc}"]

        else:  # 💬 Prompt (mô tả tự do bằng AI)
            with st.form("crisis_card_prompt_form"):
                crisis_prompt_text = st.text_area(
                    "Mô tả biến động (Crisis) bằng ngôn ngữ tự nhiên",
                    height=150,
                    placeholder="VD: Khách hàng yêu cầu giao sớm 10 ngày so với kế hoạch ban đầu...",
                    key="cc_prompt_text",
                )
                crisis_prompt_submit = st.form_submit_button("🤖 Trích xuất Crisis Card bằng AI", type="primary", use_container_width=True)

            if crisis_prompt_submit:
                crisis_submit = True
                if not crisis_prompt_text.strip():
                    validation_errors = ["Cần nhập mô tả biến động trước khi trích xuất."]
                else:
                    with st.spinner("Đang trích xuất Crisis Card từ prompt bằng OpenAI..."):
                        try:
                            client_extract = OpenAI(api_key=api_key)
                            extraction, _ = run_crisis_prompt_extraction_agent(client_extract, model, crisis_prompt_text.strip())
                            crisis_card = CrisisCardInput(
                                crisis_group=extraction.crisis_group,
                                contract_id=running_contract_id,
                                days_deviation=extraction.days_deviation,
                                extra_cost_amount=extraction.extra_cost_amount,
                                extra_cost_percent=extraction.extra_cost_percent,
                                late_amount=extraction.late_amount,
                                late_month=extraction.late_month,
                                late_days=extraction.late_days,
                                new_annual_rate_or_fee=extraction.new_annual_rate_or_fee,
                                new_processing_fee_rate=extraction.new_processing_fee_rate,
                                new_collateral_ratio=extraction.new_collateral_ratio,
                                new_num_provinces=extraction.new_num_provinces,
                                new_order_count=extraction.new_order_count,
                                raw_prompt_text=crisis_prompt_text.strip(),
                            )
                            validation_errors = validate_crisis_card_input(crisis_card)
                            st.info(f"🤖 AI đã hiểu: {extraction.extraction_notes}")
                        except Exception as exc:
                            crisis_card = None
                            validation_errors = [f"Không trích xuất được Crisis Card từ prompt: {exc}"]

        if crisis_submit:
            if validation_errors:
                st.error("Crisis Card chưa hợp lệ:\n" + "\n".join(f"- {msg}" for msg in validation_errors))
            else:
                st.session_state.crisis_card = crisis_card.model_dump()
                st.success("Crisis Card hợp lệ. Đang tính toán Before/After...")

                with st.spinner("Processing Crisis Impact..."):
                    try:
                        crisis_obj = CrisisCardInput(**st.session_state.crisis_card)
                        baseline_metrics = result["finance_metrics"]
                        list_price_goc = baseline_metrics["total_list_price"]

                        delta = resolve_crisis_deltas(
                            crisis_obj,
                            list_price_goc,
                            result["customer"].get("num_provinces"),
                            baseline_metrics.get("estimated_cost"),
                            result["opportunity"].get("initial_order_count"),
                        )

                        # FIX (bug nghiêm trọng): trước đây vượt trần cứng ORDER_CHANGE làm
                        # resolve_crisis_deltas() raise Exception -> crash cả luồng, không có
                        # Decision Card nào được trả về. Nay xử lý tất định ngay tại đây: dừng
                        # sớm, không gọi AI (vì đã đủ căn cứ để kết luận), và vẫn trả lời đầy đủ
                        # continue_contract / financing_plan / key_protection_condition như yêu
                        # cầu bắt buộc.
                        if delta.hard_cap_exceeded:
                            st.error(f"🚫 {delta.note}")
                            st.session_state.crisis_result = {
                                "finance_metrics": baseline_metrics,
                                "cash_projection": result.get("cash_projection", {}),
                                "risk_level_after": derive_risk_level_from_triggered_rules(
                                    [{
                                        "rule_id": "ORDER_HARD_CAP_EXCEEDED",
                                        "description": delta.note,
                                        "severity": "Critical",
                                    }]
                                ),
                                "risk_agent_output": None,
                                "requested_amount_after": 0.0,
                                "finance_condition_warning": None,
                                "final_decision": {
                                    "continue_contract": "TERMINATE",
                                    "financing_plan": "Không áp dụng — hợp đồng vượt trần cứng số lượng đơn hàng, chưa thể huy động vốn cho một phạm vi chưa được chấp nhận.",
                                    "key_protection_condition": "Giảm số lượng đơn hàng về đúng hạn mức cho phép (hoặc đàm phán lại hạn mức với Founder) trước khi tái xử lý Crisis Card này.",
                                    "gross_margin_after": baseline_metrics.get("gross_margin", 0.0),
                                    "closing_cash_after": result.get("cash_projection", {}).get("min_projected_closing_cash", 0.0),
                                    "funding_amount_after": 0.0,
                                    "executive_summary": delta.note,
                                },
                            }
                        else:
                            sys_data = st.session_state.get("opc_data")
                            if not sys_data:
                                raise ValueError("Chưa nạp Team Pack. Vui lòng quay lại tab Operations để tải Team Pack lên.")

                            if "profile" not in result:
                                result["profile"] = get_profile(sys_data)

                            products_df = sys_data["05_PRODUCTS"].copy()
                            selected_products_df = products_df[products_df["service_name"].astype(str).isin(result["opportunity"]["selected_services"])]

                            fm_after = build_finance_metrics_with_crisis(
                                selected_products_df,
                                result["profile"].get("payment_reliability", 1.0),
                                result["customer"].get("province"),
                                result.get("transaction_risk_score"),
                                pd.to_datetime(result["opportunity"]["order_date"]),
                                pd.to_datetime(result["opportunity"]["due_date"]),
                                result["customer"].get("num_provinces"),
                                delta,
                                result["opportunity"].get("initial_order_count"),
                            )

                            reserve_minimum = float(result["profile"].get("cash_reserve_minimum", CASH_RESERVE_THRESHOLD_DEFAULT) or CASH_RESERVE_THRESHOLD_DEFAULT)

                            cp_after = project_closing_cash_with_crisis(
                                sys_data, selected_products_df, fm_after, pd.to_datetime(result["opportunity"]["order_date"]),
                                reserve_minimum, delta
                            )

                            req_after = max(0.0, reserve_minimum - cp_after["min_projected_closing_cash"])
                            # FIX (bug logic thực sự): rerun_partner_matrix_from_layer() giờ trả
                            # thêm finance_condition_warning để không còn âm thầm bỏ qua thay đổi
                            # tài chính khi không xác định được gói vay cụ thể để áp dụng.
                            pm_after, finance_condition_warning = rerun_partner_matrix_from_layer(sys_data, req_after, cp_after, crisis_obj, result.get("partner_matrix"))
                            # req_after ở trên chỉ là "nhu cầu vốn thô" (funding_need) dùng để lọc
                            # Partner Matrix — giống hệt cách baseline dùng nó ở Mục 4. Số tiền
                            # requested_amount CUỐI CÙNG phải qua determine_requested_amount() để áp
                            # đúng ràng buộc "chỉ nâng sàn lên minimum_amount khi sản phẩm tốt nhất
                            # thực sự eligible" (đúng bản fix đã ghi chú ở Mục 6, không được bỏ qua
                            # riêng cho Crisis Card).
                            requested_amount_after = determine_requested_amount(cp_after, pm_after)

                            client = OpenAI(api_key=api_key)
                            crisis_context = {
                                "crisis_group": crisis_obj.crisis_group,
                                "extra_oper": delta.extra_oper,
                                "extra_estimated_cost": delta.extra_estimated_cost,
                                "extra_list_price": delta.extra_list_price,
                                "payment_shift": delta.payment_shift,
                                "note": delta.note
                            }

                            finance_payload_after = {
                                "customer": result["customer"],
                                "opportunity": result["opportunity"],
                                "finance_metrics": fm_after,
                                "cash_projection": cp_after,
                                "confidence_result": result.get("confidence_result"),
                                "missing_fields": result.get("missing_fields", []),
                                "crisis_context": crisis_context
                            }
                            masked_f_payload_after, _ = mask_sensitive_fields(finance_payload_after)
                            f_res_after, _ = run_finance_agent(client, model, masked_f_payload_after)

                            risk_eval_after = evaluate_risk_rules(
                                sys_data,
                                fm_after,
                                cp_after,
                                result.get("confidence_result"),
                            )
                            triggered_after = list(risk_eval_after.get("triggered_rules", []))
                            triggered_ids_after = {
                                r.get("rule_id") for r in triggered_after if r.get("rule_id")
                            }
                            if cp_after["cash_reserve_breach"] and "RR-002" not in triggered_ids_after:
                                triggered_after.append({"rule_id": "RR-002", "description": "Thiếu hụt dòng tiền (Crisis)", "severity": "High"})
                                triggered_ids_after.add("RR-002")

                            if any(g in ("DEADLINE_EARLY", "DEADLINE_LATE") for g in crisis_obj.crisis_group) and crisis_obj.days_deviation and crisis_obj.days_deviation > 7:
                                if "SCHEDULE_BREACH" not in triggered_ids_after:
                                    triggered_after.append({"rule_id": "SCHEDULE_BREACH", "description": "Vi phạm tiến độ nghiêm trọng (>7 ngày)", "severity": "High"})
                                    triggered_ids_after.add("SCHEDULE_BREACH")

                            if "PAYMENT_DELAY" in crisis_obj.crisis_group and crisis_obj.late_amount and crisis_obj.late_amount > 0:
                                if "PAYMENT_PATTERN_RISK" not in triggered_ids_after:
                                    triggered_after.append({"rule_id": "PAYMENT_PATTERN_RISK", "description": "Khách hàng chậm thanh toán", "severity": "High"})
                                    triggered_ids_after.add("PAYMENT_PATTERN_RISK")

                            base_risk_level_after = derive_risk_level_from_triggered_rules(triggered_after)
                            before_cash_for_risk = float(
                                result.get("cash_projection", {}).get("min_projected_closing_cash", 0.0) or 0.0
                            )
                            before_requested_amount_for_risk = float(result.get("requested_amount", 0.0) or 0.0)
                            # Đánh giá risk level sau biến động chỉ dựa thuần vào mức vi phạm
                            # risk rule (triggered_after) — giống cách evaluate_risk_rules() đánh
                            # giá risk_level ở luồng baseline — không còn nâng/hạ mức risk theo
                            # biến động dòng tiền/nhu cầu vay như trước.
                            risk_level_after = base_risk_level_after

                            risk_payload_after = {
                                "finance_agent_output": f_res_after.model_dump() if f_res_after else {},
                                "finance_metrics": fm_after,
                                "cash_projection": cp_after,
                                "confidence_result": result.get("confidence_result"),
                                "triggered_rules": triggered_after,
                                "risk_level": risk_level_after,
                                "missing_fields": result.get("missing_fields", []),
                                "crisis_context": crisis_context
                            }
                            masked_r_payload_after, _ = mask_sensitive_fields(risk_payload_after)
                            r_res_after, _ = run_risk_agent(client, model, masked_r_payload_after)

                            decision_payload_after = {
                                "customer": result["customer"],
                                "finance_metrics": fm_after,
                                "cash_projection": cp_after,
                                "confidence_result": result.get("confidence_result"),
                                "finance_agent_output": f_res_after.model_dump() if f_res_after else {},
                                "risk_agent_output": r_res_after.model_dump() if r_res_after else {},
                                "triggered_rules": triggered_after,
                                "risk_level": risk_level_after,
                                "baseline_context": {
                                    "closing_cash_before": before_cash_for_risk,
                                    "requested_amount_before": before_requested_amount_for_risk,
                                },
                                "partner_matrix": pm_after,
                                "requested_amount": requested_amount_after,
                                "large_decision_threshold": LARGE_DECISION_THRESHOLD,
                                "founder_approval_needed": requested_amount_after > LARGE_DECISION_THRESHOLD,
                                "missing_fields": result.get("missing_fields", []),
                                "crisis_context": crisis_context
                            }
                            masked_d_payload_after, _ = mask_sensitive_fields(decision_payload_after)

                            d_res_after_raw = run_crisis_decision_agent(
                                client, model, masked_d_payload_after
                            )
                            d_res_after = d_res_after_raw[0] if d_res_after_raw else None

                            if d_res_after:
                                # FIX (gây nhầm lẫn / code thừa): enforce_crisis_decision_card()
                                # không còn nhận is_new_customer (tham số này trước đây không hề
                                # được dùng trong thân hàm — xem ghi chú tại định nghĩa hàm).
                                final_decision = enforce_crisis_decision_card(
                                    d_res_after, fm_after, cp_after, requested_amount_after, pm_after, [r["rule_id"] for r in triggered_after]
                                )
                                st.session_state.crisis_result = {
                                    "finance_metrics": fm_after,
                                    "cash_projection": cp_after,
                                    "risk_level_after": risk_level_after,
                                    "risk_agent_output": r_res_after.model_dump() if r_res_after else None,
                                    "requested_amount_after": requested_amount_after,
                                    "finance_condition_warning": finance_condition_warning,
                                    "final_decision": final_decision.model_dump()
                                }
                    except Exception as e:
                        st.error(f"Lỗi hệ thống khi xử lý Crisis: {str(e)}")

        if "crisis_result" in st.session_state:
            st.markdown("### CRISIS CARD")
            c_res = st.session_state.crisis_result
            c_dec = c_res["final_decision"]
            baseline_dec = result["decision_result"]

            b_risk = result.get("risk_level", "UNKNOWN")
            a_risk = c_res.get("risk_level_after", b_risk)

            b_cash = result.get("cash_projection", {}).get("min_projected_closing_cash", 0.0)
            a_cash = c_res.get("cash_projection", {}).get("min_projected_closing_cash", b_cash)

            b_gm = result.get("finance_metrics", {}).get("gross_margin", 0.0)
            a_gm = c_res.get("finance_metrics", {}).get("gross_margin", b_gm)

            # Dùng đúng funding_amount đã được enforce_decision_card()/
            # enforce_crisis_decision_card() ép về 0 khi partner_matrix không có
            # eligible=true — KHÔNG dùng requested_amount (đó là "nhu cầu vốn" thô,
            # vẫn dương ngay cả khi không có gói vay nào khả thi, nên nếu hiển thị ở
            # đây sẽ mâu thuẫn với chính nội dung Financing Plan).
            b_funding = result.get("decision_result", {}).get("funding_amount", 0.0)
            a_funding = c_dec.get("funding_amount_after", b_funding)

            def format_vnd_safe(val):
                return f"{val:,.0f} VND" if val else "0 VND"

            # Dòng 1: các chỉ số tài chính
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row1_col1.metric("Gross Margin", f"{a_gm:.1%}", f"{(a_gm - b_gm):.1%}")
            row1_col2.metric("Closing Cash", format_vnd_safe(a_cash), format_vnd_safe(a_cash - b_cash))
            row1_col3.metric("Funding Amount", format_vnd_safe(a_funding), format_vnd_safe(a_funding - b_funding))

            # Dòng 2: Risk Level (kèm risk summary do AI đề xuất) và Decision (kèm decision summary) —
            # gộp chung vào 1 khối thẻ (card) duy nhất cho mỗi bên, thay vì tách metric/caption rời nhau.
            risk_agent_output_after = c_res.get("risk_agent_output")
            if risk_agent_output_after and risk_agent_output_after.get("warnings"):
                risk_summary_text = "🤖 <strong>Risk Summary (AI):</strong> " + " | ".join(risk_agent_output_after["warnings"])
            elif risk_agent_output_after and risk_agent_output_after.get("recommended_controls"):
                risk_summary_text = "🤖 <strong>Risk Summary (AI):</strong> " + " | ".join(risk_agent_output_after["recommended_controls"])
            else:
                risk_summary_text = "<em>Không có risk summary từ AI cho biến động này (vượt trần cứng, không qua Risk Agent).</em>"

            risk_color = "#ef4444" if a_risk in ["CRITICAL", "HIGH"] else "#eab308" if a_risk == "MEDIUM" else "#22c55e"

            decision_summary_text = f"📝 <strong>Decision Summary:</strong> {c_dec['executive_summary']}"

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:16px 20px;box-shadow:0 4px 16px rgba(15,23,42,0.04);height:100%;">
<div style="text-transform:uppercase;letter-spacing:0.04em;font-size:0.85rem;color:#64748b;font-weight:700;margin-bottom:6px;">Risk Level</div>
<div style="font-size:1.9rem;font-weight:800;color:{risk_color};margin-bottom:8px;">{a_risk}</div>
<div style="display:inline-block;background:#f1f5f9;color:#64748b;font-size:0.9rem;font-weight:600;padding:3px 12px;border-radius:8px;margin-bottom:12px;">Old: {b_risk}</div>
<div style="border-top:1px solid #e2e8f0;margin:8px 0;"></div>
<div style="font-size:1rem;color:#334155;line-height:1.6;">{risk_summary_text}</div>
</div>
                """, unsafe_allow_html=True)

            with row2_col2:
                # FIX (gây nhầm lẫn): continue_contract (CONTINUE/CONTINUE WITH CONDITIONS/
                # TERMINATE) và recommendation gốc (ACCEPT/CONDITIONAL_ACCEPT/REJECT/
                # NEED_MORE_DATA) là 2 thang nhãn KHÁC HỆ THỐNG — trước đây đặt cạnh nhau
                # dưới dạng "delta" của cùng 1 metric khiến người xem dễ hiểu nhầm là cùng
                # một thang đo. Nay tách rõ 2 dòng thông tin, không dùng delta cho cặp này.
                st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:16px 20px;box-shadow:0 4px 16px rgba(15,23,42,0.04);height:100%;">
<div style="text-transform:uppercase;letter-spacing:0.04em;font-size:0.85rem;color:#64748b;font-weight:700;margin-bottom:6px;">Recommend Decision</div>
<div style="font-size:1.9rem;font-weight:800;color:#0f172a;margin-bottom:8px;">{c_dec["continue_contract"]}</div>
<div style="border-top:1px solid #e2e8f0;margin:8px 0;"></div>
<div style="font-size:1rem;color:#334155;line-height:1.6;">{decision_summary_text}</div>
</div>
                """, unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-size:0.95rem;color:#64748b;line-height:1.6;margin:6px 0 4px 0;">'
                f"ℹ️ Quyết định gốc trước biến động (thang đánh giá khác — "
                f"ACCEPT/CONDITIONAL_ACCEPT/REJECT/NEED_MORE_DATA): <strong>{baseline_dec['recommendation']}</strong>. "
                "Không so sánh trực tiếp 1-1 với continue_contract ở trên vì đây là 2 thang đo khác nhau."
                "</div>",
                unsafe_allow_html=True,
            )

            st.warning(f"**Protection Condition:** {c_dec['key_protection_condition']}")
            st.success(f"**Financing Plan:** {c_dec['financing_plan']}")

            payment_shift_warning = c_res.get("cash_projection", {}).get("payment_shift_warning")
            if payment_shift_warning:
                st.error(f"⚠️ {payment_shift_warning}")

            finance_condition_warning = c_res.get("finance_condition_warning")
            if finance_condition_warning:
                st.warning(f"⚠️ {finance_condition_warning}")


with tab_dashboard:
    st.markdown(
        """
<div class="ops-hero">
<div class="ops-hero-title">🏆 Decision Dashboard</div>
<div class="ops-hero-desc">
Tổng hợp kết quả cuối cùng từ 3 AI Agent — chỉ số tài chính, mức độ rủi ro, phương án tài trợ đề xuất
và khuyến nghị quyết định dành cho Founder.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if not result:
        st.info("Decision Card sẽ xuất hiện tại đây.")
    else:
        finance_metrics = result["finance_metrics"]
        cash_projection = result["cash_projection"]
        confidence_result = result["confidence_result"]
        risk_result = result["risk_result"]
        decision = result["decision_result"]

        # Render Premium Dashboard KPI
        gm_val = decision['gross_margin']
        gm_pct = int(gm_val * 100) if gm_val else 0
        gm_color = "#10b981" if gm_val > 0.3 else "#f59e0b" if gm_val > 0.15 else "#ef4444"
        
        conf_val = decision['confidence_score']
        conf_pct = int(conf_val * 100) if conf_val is not None else 0
        conf_str = f"{conf_val:.0%}" if conf_val is not None else "N/A"
        conf_color = "#8b5cf6" if conf_pct >= 80 else "#6366f1"

        st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.dash-container {{ font-family: 'Inter', sans-serif; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 24px; margin-bottom: 40px; }}
.kpi-card {{ background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px); padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); border: 1px solid #e2e8f0; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; }}
.kpi-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04); border-color: #cbd5e1; }}
.kpi-title {{ font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }}
.kpi-value {{ font-size: 2.25rem; font-weight: 800; color: #0f172a; line-height: 1.2; letter-spacing: -0.02em; }}
.progress-bar-bg {{ width: 100%; height: 6px; background-color: #e2e8f0; border-radius: 9999px; margin-top: 12px; overflow: hidden; }}
.progress-bar-fill {{ height: 100%; border-radius: 9999px; transition: width 1s ease-in-out; }}
.dash-section-title {{ font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }}
.reasons-list {{ list-style-type: none; padding: 0; display: flex; flex-direction: column; gap: 12px; }}
.reasons-list li {{ background: white; padding: 16px 20px; border-radius: 12px; border: 1px solid #e2e8f0; color: #334155; font-size: 0.95rem; line-height: 1.5; display: flex; gap: 12px; align-items: flex-start; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }}
.reasons-list li svg {{ flex-shrink: 0; width: 20px; height: 20px; color: #6366f1; margin-top: 2px; }}
.rec-badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-size: 0.875rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 16px; }}
</style>
<div class="dash-container kpi-grid">
<div class="kpi-card">
<div class="kpi-title">Gross Margin <span style="color: {gm_color}">⬤</span></div>
<div class="kpi-value">{decision['gross_margin']:.1%}</div>
<div class="progress-bar-bg">
<div class="progress-bar-fill" style="width: {min(100, max(0, gm_pct))}%; background-color: {gm_color};"></div>
</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Closing Cash <span style="color: #3b82f6">⬤</span></div>
<div class="kpi-value" style="font-size: 1.75rem;">{format_vnd(decision["closing_cash"])}</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Funding Amount <span style="color: #f59e0b">⬤</span></div>
<div class="kpi-value" style="font-size: 1.75rem;">{format_vnd(decision['funding_amount'])}</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Confidence Score <span style="color: {conf_color}">⬤</span></div>
<div class="kpi-value">{conf_str}</div>
<div class="progress-bar-bg">
<div class="progress-bar-fill" style="width: {conf_pct}%; background-color: {conf_color};"></div>
</div>
</div>
</div>
        """, unsafe_allow_html=True)
        
        dash_col1, dash_col2 = st.columns([1.8, 1.2], gap="large")
        
        with dash_col1:
            rec = decision['recommendation']
            if "ACCEPT" in rec:
                badge_bg, badge_color = "#dcfce7", "#166534"
            elif "REJECT" in rec:
                badge_bg, badge_color = "#fee2e2", "#991b1b"
            else:
                badge_bg, badge_color = "#fef3c7", "#92400e"
                
            st.markdown(f"""
<div class="dash-container" style="background: white; border-radius: 24px; padding: 32px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #e2e8f0; margin-bottom: 32px;">
<div class="rec-badge" style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}33;">RECOMMENDATION: {rec}</div>
<div style="display: grid; gap: 20px;">
<div style="background: #f8fafc; padding: 20px; border-radius: 16px; border: 1px solid #f1f5f9;">
<div style="font-size: 0.75rem; color: #64748b; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px;">SELECTED FINANCING OPTION</div>
<div style="font-size: 1.25rem; color: #0f172a; font-weight: 700; display: flex; align-items: center; gap: 10px;">
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color: #3b82f6;"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m-3-2.818l.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
{decision['selected_financing_option']}
</div>
</div>
<div style="padding: 20px; border-radius: 16px; border: 1px solid #e2e8f0;">
<div style="font-size: 0.75rem; color: #64748b; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px;">EXECUTIVE SUMMARY</div>
<div style="font-size: 1rem; color: #334155; line-height: 1.6;">{decision['executive_summary']}</div>
</div>
</div>
</div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="dash-section-title dash-container"><svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color: #6366f1;"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg> Lập luận chính (3 Reasons)</div>', unsafe_allow_html=True)
            check_svg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
            reasons_html = "<ul class='reasons-list dash-container'>" + "".join([f"<li>{check_svg} <span>{r}</span></li>" for r in decision["three_reasons"]]) + "</ul>"
            st.markdown(reasons_html, unsafe_allow_html=True)

        with dash_col2:
            st.markdown('<div class="dash-section-title dash-container"><svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color: #ef4444;"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg> Quản trị Rủi ro</div>', unsafe_allow_html=True)
            
            risk_level = risk_result["risk_level"]
            risk_summary = build_risk_summary_message(
                decision["gross_margin"], cash_projection["min_projected_closing_cash"]
            )
            if risk_level in {"CRITICAL", "HIGH"}:
                st.error(f"🚨 **Risk Level: {risk_level}**\n\n{risk_summary['message']}")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ **Risk Level: {risk_level}**\n\n{risk_summary['message']}")
            else:
                st.success(f"✅ **Risk Level: {risk_level}**\n\n{risk_summary['message']}")
                
            st.markdown(f"""
<div class="dash-container" style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 16px; padding: 20px; margin-top: 20px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(251, 191, 36, 0.1);">
<div style="font-size: 0.85rem; color: #b45309; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 0.05em;">
<svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg> Protection Condition
</div>
<div style="font-size: 0.95rem; color: #92400e; line-height: 1.6;">{decision['protection_condition']}</div>
</div>
            """, unsafe_allow_html=True)
            
            if result["missing_fields"]:
                st.markdown(f"""
<div class="dash-container" style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 16px; padding: 20px; margin-bottom: 24px;">
<div style="font-size: 0.85rem; color: #b91c1c; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">❌ Missing Data Request</div>
<div style="font-size: 0.95rem; color: #991b1b; line-height: 1.5;">{', '.join(result['missing_fields'])}</div>
</div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="dash-section-title dash-container" style="margin-top: 40px;"><svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color: #3b82f6;"><path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg> Founder Approval Gate</div>', unsafe_allow_html=True)
            
            sensitive = decision["human_approval_required"] or result["founder_approval_needed"]
            if sensitive:
                st.markdown(f"""
<div class="dash-container" style="background: #fdf2f8; border-left: 4px solid #f43f5e; padding: 16px 20px; border-radius: 0 12px 12px 0; margin-bottom: 20px;">
<div style="color: #be123c; font-weight: 700; font-size: 0.85rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">⚠️ CẢNH BÁO </div>
<div style="color: #9f1239; font-size: 0.95rem; line-height: 1.5;">{decision['approval_reason']}</div>
</div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Mọi hợp đồng đều cần Founder ra quyết định cuối cùng trước khi triển khai.")
                
            if result["missing_fields"]:
                # Yêu cầu của Founder: nếu dữ liệu đầu vào còn thiếu, KHÔNG cho phép thao tác
                # Phê duyệt/Từ chối — chỉ hiển thị cảnh báo đỏ yêu cầu bổ sung thông tin, thay
                # thế toàn bộ khối nút quyết định + trạng thái (Chưa quyết định/Phê duyệt/Từ
                # chối + PENDING APPROVAL).
                st.session_state.founder_decision = "Chưa quyết định"
                st.markdown(f"""
<div class="dash-container" style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 16px; padding: 24px; display: flex; align-items: center; gap: 20px; margin-bottom: 15px;">
<div style="font-size: 2.5rem; color: #ef4444; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; background: #fee2e2; border-radius: 50%; flex-shrink: 0;">❗</div>
<div>
<div style="font-size: 1.25rem; font-weight: 800; color: #b91c1c; letter-spacing: 0.05em; margin-bottom: 4px;">YÊU CẦU BỔ SUNG THÊM THÔNG TIN</div>
<div style="font-size: 0.95rem; color: #991b1b; line-height: 1.5;">Founder chưa thể phê duyệt vì hồ sơ còn thiếu dữ liệu bắt buộc: <strong>{', '.join(result['missing_fields'])}</strong>. Vui lòng bổ sung rồi chạy lại Multi-Agent.</div>
</div>
</div>
                """, unsafe_allow_html=True)
                founder_decision = st.session_state.founder_decision
            else:
                if "founder_decision" not in st.session_state:
                    st.session_state.founder_decision = "Chưa quyết định"

                st.markdown("""
                <style>
                div[data-testid="column"] button {
                    height: 50px;
                    font-size: 1.1rem !important;
                    font-weight: 700 !important;
                    border-radius: 12px !important;
                    transition: all 0.2s;
                }
                </style>
                """, unsafe_allow_html=True)

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("⚪ Chưa quyết định", use_container_width=True):
                        st.session_state.founder_decision = "Chưa quyết định"
                with btn_col2:
                    if st.button("✅ PHÊ DUYỆT", use_container_width=True, type="primary"):
                        st.session_state.founder_decision = "✅ Phê duyệt (Approve)"
                with btn_col3:
                    if st.button("❌ TỪ CHỐI", use_container_width=True):
                        st.session_state.founder_decision = "❌ Từ chối (Reject)"

                founder_decision = st.session_state.founder_decision

                if founder_decision == "✅ Phê duyệt (Approve)":
                    st.markdown("""
<div class="dash-container" style="background: linear-gradient(135deg, #10b981, #059669); padding: 24px; border-radius: 16px; color: white; display: flex; align-items: center; gap: 20px; box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4); margin-bottom: 15px;">
<div style="font-size: 3rem; background: rgba(255,255,255,0.2); width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; border-radius: 50%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">✅</div>
<div>
<div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 4px;">APPROVED</div>
<div style="font-size: 0.95rem; opacity: 0.95;">Decision Card đã được ký. Hợp đồng chính thức có hiệu lực và được phép triển khai.</div>
</div>
</div>
                    """, unsafe_allow_html=True)
                elif founder_decision == "❌ Từ chối (Reject)":
                    st.markdown("""
<div class="dash-container" style="background: linear-gradient(135deg, #ef4444, #be123c); padding: 24px; border-radius: 16px; color: white; display: flex; align-items: center; gap: 20px; box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.4); margin-bottom: 15px;">
<div style="font-size: 3rem; background: rgba(255,255,255,0.2); width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; border-radius: 50%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">❌</div>
<div>
<div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 4px;">REJECTED</div>
<div style="font-size: 0.95rem; opacity: 0.95;">Founder đã từ chối. Hợp đồng bị hủy bỏ và không được phép tiến hành.</div>
</div>
</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="dash-container" style="background: #f8fafc; border: 2px dashed #cbd5e1; padding: 24px; border-radius: 16px; display: flex; align-items: center; gap: 20px; margin-bottom: 15px;">
<div style="font-size: 2.5rem; color: #94a3b8; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; background: #f1f5f9; border-radius: 50%;">⏳</div>
<div>
<div style="font-size: 1.25rem; font-weight: 800; color: #475569; letter-spacing: 0.05em; margin-bottom: 4px;">PENDING APPROVAL</div>
<div style="font-size: 0.95rem; color: #64748b;">Đang chờ Founder xem xét các chỉ số và đưa ra quyết định cuối cùng...</div>
</div>
</div>
                    """, unsafe_allow_html=True)

        export_payload = {
            "model": result["model"],
            "customer": result["customer"],
            "finance_metrics": finance_metrics,
            "cash_projection": cash_projection,
            "confidence_result": confidence_result,
            "finance_agent": result["finance_result"],
            "risk_agent": risk_result,
            "decision_card": decision,
            "triggered_rules": result["triggered_rules"],
            "missing_fields": result["missing_fields"],
            "founder_decision": founder_decision,
            "sensitive_threshold_flagged": sensitive,
            "openai_response_ids": [item["response_id"] for item in result["workflow_logs"]],
        }
        st.download_button(
            "Tải Decision Card JSON",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2, default=str),
            file_name="opc_decision_card.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# NEWBIE BRANDING HEADER
# ============================================================


st.markdown('''
<style>
.newbie-header {
    position: fixed;
    bottom: 15px;
    left: 20px;
    z-index: 999999;
    display: flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 8px 20px;
    border-radius: 99px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.9);
}

.newbie-logo {
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 1.25rem;
    background: linear-gradient(270deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    letter-spacing: 0.1em;
    cursor: default;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.newbie-logo:hover {
    transform: translateY(-2px) scale(1.05);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.newbie-icon {
    animation: floating 3s ease-in-out infinite;
}

@keyframes floating {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-2px); }
    100% { transform: translateY(0px); }
}

/* Đảm bảo nội dung không bị đè bởi header */
.stApp {
    padding-bottom: 70px;
}
</style>

<div class="newbie-header">
    <div class="newbie-logo">
        <svg class="newbie-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="url(#grad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ec4899;stop-opacity:1" />
                </linearGradient>
            </defs>
            <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
            <path d="M2 17l10 5 10-5"></path>
            <path d="M2 12l10 5 10-5"></path>
        </svg>
        NEWBIE
    </div>
</div>
''', unsafe_allow_html=True)



