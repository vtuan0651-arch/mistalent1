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
OPENAI_API_KEY_HARDCODED = ""  # <-- dán OpenAI API key của bạn vào đây nếu muốn

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

# Customer engagement = (Điểm lịch sử giao dịch) + (Điểm pricing model)
#   - Điểm lịch sử giao dịch: >3 giao dịch = 0.5; 1-2 giao dịch = 0.25; 0 giao dịch = 0
#   - Điểm pricing model: Initial setup = 0.5; Monthly subscription = 0.25; Project = 0.2
PRICING_MODEL_ENGAGEMENT_SCORE = {
    PRICING_MODEL_INITIAL_SETUP: 0.5,
    PRICING_MODEL_MONTHLY_SUBSCRIPTION: 0.25,
    PRICING_MODEL_PROJECT: 0.2,
}


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


def compute_oper_coefficient(
    payment_reliability: Optional[float],
    province: Optional[str],
    transaction_risk_score: Optional[float],
    order_date: pd.Timestamp,
    due_date: pd.Timestamp,
) -> tuple[float, list[dict]]:
    """Cộng dồn hệ số Oper theo bảng điều kiện của System Prompt.

    Ghi chú: Uy tín thanh toán (Payment Reliability) và Áp lực tiến độ giao hàng
    (Urgent Delivery) đã được thay thế bằng một hệ số rủi ro con người CỐ ĐỊNH
    +4.0%, luôn được cộng vào Oper bất kể payment_reliability / thời hạn hợp đồng.
    """
    oper = 0.0
    breakdown = []

    # Hệ số rủi ro con người (Human Risk Factor) — cố định, luôn áp dụng.
    oper += 0.04
    breakdown.append({"tieu_chi": "Rủi ro con người (Human Risk Factor - cố định)", "he_so": 0.04})

    if province is not None and not is_core_city(province):
        oper += 0.03
        breakdown.append({"tieu_chi": f"Mở rộng địa bàn ({province})", "he_so": 0.03})

    if transaction_risk_score is not None and transaction_risk_score > 85:
        oper += 0.04
        breakdown.append({"tieu_chi": "Rủi ro giao dịch > 85", "he_so": 0.04})

    return oper, breakdown


def build_finance_metrics(
    selected_products: pd.DataFrame,
    payment_reliability: Optional[float],
    province: Optional[str],
    transaction_risk_score: Optional[float],
    order_date: pd.Timestamp,
    due_date: pd.Timestamp,
) -> dict:
    """
    baseline_estimate = Σ (list_price × (1 - target_margin))
    estimated_cost   = baseline_estimate × (1 + Oper)
    Gross_Margin     = (Σ list_price - estimated_cost) / Σ list_price
    """
    total_list_price = float(selected_products["list_price"].sum())
    baseline_estimate = float(
        (selected_products["list_price"] * (1 - selected_products["target_margin"])).sum()
    )

    oper, oper_breakdown = compute_oper_coefficient(
        payment_reliability, province, transaction_risk_score, order_date, due_date
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
# Chỉ tính Confidence Score SAU KHI đã lọc 3 lớp sản phẩm ngân hàng (account_ops /
# credit_guarantee / unclassified bị loại) và thu được ÍT NHẤT một gói vay phù hợp
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
    """Chỉ tính Confidence Score khi (sau khi lọc 3 lớp) có ít nhất một gói vay
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
    """Phân loại 1 sản phẩm 11_BANK_PRODUCTS theo bản chất, trả về (category, lý do).

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
    """Chỉ truy xuất 11_BANK_PRODUCTS khi Projected_Closing_Cash < 550 triệu VND.
    So sánh các gói TÍN DỤNG BƠM TIỀN MẶT TRỰC TIẾP (category="credit_cash") —
    không lọc theo target_segment/customer_type, nhưng loại bỏ:
      (1) dịch vụ vận hành tài khoản (account_ops) — không phải khoản vay;
      (2) sản phẩm tín dụng bảo lãnh/hỗ trợ giao dịch (credit_guarantee) — không
          bơm tiền mặt trực tiếp nên sai mục đích so với nhu cầu bù đắp RR-002;
      (3) sản phẩm chưa phân loại được (unclassified) — không tự đoán, xem
          classify_all_bank_products() để Founder tự rà soát."""
    if not cash_projection["cash_reserve_breach"]:
        return []

    candidates = data["11_BANK_PRODUCTS"].copy()

    matrix = []
    for _, product in candidates.iterrows():
        category, _reason = classify_bank_product(
            str(product["product_name"]), str(product.get("description", ""))
        )
        if category != "credit_cash":
            continue

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

    # Sắp xếp: ưu tiên eligible, sau đó chi phí thấp nhất, rồi tỉ lệ bảo đảm thấp nhất.
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
  preliminary_assessment phải là NEED_MORE_DATA.
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
- risk_level phải khớp với risk_level đã được Python tính (dùng lại nguyên giá trị).
- Nếu missing_fields không rỗng, phải nêu yêu cầu bổ sung dữ liệu.
- Viết bằng tiếng Việt, ngắn gọn và có thể hành động.
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
3. Đưa ra recommendation phục vụ Founder.

Quy tắc bắt buộc:
- Nếu missing_fields không rỗng: recommendation=NEED_MORE_DATA.
- gross_margin, closing_cash, confidence_score PHẢI lấy đúng giá trị Python cung cấp,
  không tự tính lại.
- Chỉ chọn phương án tài chính có eligible=true trong partner_matrix; nếu partner_matrix
  rỗng (không cần vay), selected_financing_option phải nêu rõ "Không cần huy động vốn ngoài".
- Nếu requested_amount > 300,000,000 VND: human_approval_required=true (RR-005, Founder phê duyệt).
- Không phát minh sản phẩm, lãi suất hoặc hạn mức ngoài dữ liệu được cung cấp.
- three_reasons phải có chính xác 3 phần tử, MỖI phần tử đánh giá đúng 1 trong 3 chỉ số
  bắt buộc theo thứ tự cố định:
      (1) Lý do về gross_margin — đối chiếu RR-003 (gross_margin < 0.28) nếu bị kích hoạt,
      nếu không kích hoạt vẫn phải nhận xét gross_margin đang ở mức an toàn hay không.
      (2) Lý do về closing_cash — đối chiếu RR-002 (Projected_Closing_Cash < 550 triệu VND)
      nếu bị kích hoạt, nếu không kích hoạt vẫn phải nhận xét khả năng thanh khoản.
      (3) Lý do về confidence_score — đối chiếu RR-006 (confidence_score < 0.65, chỉ áp dụng
      khi RR-002 đã kích hoạt) nếu bị kích hoạt, nếu không kích hoạt (hoặc confidence_score
      là None) vẫn phải nhận xét mức độ tin cậy dữ liệu.
      Mỗi lý do phải nêu rõ số liệu cụ thể (giá trị chỉ số) và rule liên quan nếu có kích hoạt,
      không được viết chung chung hay gộp nhiều chỉ số vào 1 lý do.
- protection_condition phải là một điều kiện thương mại hoặc kiểm soát cụ thể cần Founder xác nhận.
- Viết bằng tiếng Việt, rõ ràng và bảo vệ được khi vấn đáp.
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
            "Giải ngân/triển khai hợp đồng theo tiến độ từng giai đoạn (phase delivery), "
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
    # được OpenAI tự ý bỏ qua. Nếu OpenAI tự đánh giá cần duyệt vì lý do khác, vẫn giữ.
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
# 8. UI
# ============================================================

st.markdown(
    """
<style>
/* Import font hiện đại */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Nền tảng tổng thể - Gradient nhẹ nhàng */
.stApp {
    background: linear-gradient(135deg, #f0f4fd 0%, #ffffff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 95% !important;
}

/* Hiệu ứng nổi bồng bềnh cho các thành phần */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}

/* Agent Card - Glassmorphism & Hover */
.agent-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
}

.agent-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(43, 88, 255, 0.08);
    border-color: rgba(43, 88, 255, 0.2);
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
    padding: 0.5rem 1rem;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}

.stButton > button:active {
    transform: translateY(1px);
}

/* Inputs & Selectboxes */
.stTextInput > div > div > input, 
.stSelectbox > div > div > div {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    background-color: #ffffff !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1e3a8a !important;
}

div[data-testid="stMetricLabel"] {
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.85rem !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    border: 1px solid #f1f5f9;
}

/* Muted text */
.small-muted {
    font-size: 0.85rem; 
    color: #94a3b8;
    font-weight: 400;
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
<div style="text-align: center; margin-top: 10px; margin-bottom: 30px; position: relative; z-index: 50;">
    <div style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 12px 36px;
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
            font-size: 2rem;
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
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 20px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    ">
        Team Pack (CSV) <span style="color: #3b82f6">→</span> Finance Agent <span style="color: #3b82f6">→</span> Risk Agent <span style="color: #3b82f6">→</span> Decision Agent <span style="color: #3b82f6">→</span> Founder Approval
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Cấu hình")
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
        value="gpt-4o-mini",
        key="model_input_v4",
        help="Model OpenAI hỗ trợ Structured Outputs, ví dụ: gpt-4o-mini, gpt-4o, gpt-4.1, gpt-4.1-mini.",
    )
    st.info("API key không được ghi vào Excel, prompt log hoặc Decision Card.")

result = st.session_state.get("opc_result")

st.markdown("""
<style>
/* Khung bao quanh thanh tab chính, giúp tab nổi bật và rõ ràng hơn */
div[data-testid="stTabs"] {
    background: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 10px 14px 0 14px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    height: 56px;
    padding: 0 22px;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #475569;
    border-radius: 10px 10px 0 0;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] p {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #b91c1c !important;
    background: #ffffff;
    border-bottom: 4px solid #ef4444 !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #ef4444 !important;
    height: 4px !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

tab_ops, tab_dashboard = st.tabs(["⚙️ Operations (Input & Workflow)", "🏆 Decision Dashboard"])

with tab_ops:
    col_input, col_workflow = st.columns([1.0, 2.2], gap="large")

    with col_input:
        st.subheader("1. Input Data")
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

                        # Lọc 3 lớp sản phẩm ngân hàng (account_ops / credit_guarantee /
                        # unclassified bị loại) và xác định gói vay đề xuất TRƯỚC KHI tính
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
                                },
                                "finance_metrics": finance_metrics,
                                "cash_projection": cash_projection,
                                "confidence_result": confidence_result,
                                "missing_fields": missing_fields,
                            }
                            
                            finance_result, finance_response_id = run_finance_agent(
                                client, model, finance_payload
                            )
                                
                            workflow_logs.append(
                                {
                                    "agent": "Data & Finance Agent",
                                    "response_id": finance_response_id,
                                    "result": finance_result.model_dump(),
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
                            
                            risk_result, risk_response_id = run_risk_agent(client, model, risk_payload)
                                
                            workflow_logs.append(
                                {
                                    "agent": "Risk & Compliance Agent",
                                    "response_id": risk_response_id,
                                    "result": risk_result.model_dump(),
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
                            
                            decision_result, decision_response_id = run_decision_agent(
                                client, model, decision_payload
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
                        st.session_state["opc_result"] = {
                            "model": model,
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
                            },
                            "finance_metrics": finance_metrics,
                            "cash_projection": cash_projection,
                            "confidence_result": confidence_result,
                            "missing_fields": missing_fields,
                            "triggered_rules": risk_eval["triggered_rules"],
                            "risk_level": risk_eval["risk_level"],
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
        st.subheader("2. Agent Workflow")
        if not result:
            st.info("Workflow sẽ xuất hiện sau khi chạy hệ thống.")
        else:
            st.success(
                f"OpenAI API • Model: {result['model']} • "
                f"{result['elapsed_seconds']:.1f}s"
            )

            for index, log in enumerate(result["workflow_logs"], start=1):
                with st.expander(f"{index}. {log['agent']} — Completed", expanded=True):
                    st.caption("OpenAI response ID: " + str(log["response_id"]))
                    res = log["result"]
                    agent_name = log["agent"]

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


with tab_dashboard:
    st.subheader("3. Decision Dashboard")
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
            if risk_level in {"CRITICAL", "HIGH"}:
                st.error(f"🚨 **Risk Level: {risk_level}**\n\nCần đặc biệt lưu ý và kiểm soát nghiêm ngặt.")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ **Risk Level: {risk_level}**\n\nRủi ro có thể chấp nhận nếu tuân thủ điều kiện bảo vệ.")
            else:
                st.success(f"✅ **Risk Level: {risk_level}**\n\nHợp đồng ở ngưỡng an toàn.")
                
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
st.markdown("""
<style>
.fullscreen-btn {
    position: fixed;
    bottom: 15px;
    right: 20px;
    z-index: 999999;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.9rem;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}
.fullscreen-btn:hover {
    transform: translateY(-2px);
}
</style>
<button class="fullscreen-btn" onclick="
    var elem = document.documentElement;
    if (!document.fullscreenElement) {
        elem.requestFullscreen().catch(function(err) {
            alert('Không thể bật toàn màn hình: ' + err.message);
        });
    } else {
        document.exitFullscreen();
    }
">⛶ Toàn màn hình</button>
""", unsafe_allow_html=True)

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
