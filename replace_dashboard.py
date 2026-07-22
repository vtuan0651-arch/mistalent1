import os

with open("g:/OneDrive - Đại học Thương mại/Máy tính/MIS/app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_block = """        metric_cols = st.columns(3)
        metric_cols[0].metric("Gross Margin", f"{decision['gross_margin']:.1%}")
        metric_cols[1].metric("Closing Cash", format_vnd(decision["closing_cash"]))
        metric_cols[2].metric(
            "Confidence Score",
            f"{decision['confidence_score']:.0%}" if decision["confidence_score"] is not None else "N/A",
        )

        st.markdown(
            f\"\"\"
<div class="decision-card">
<b>Recommendation</b><br>
<h2>{decision['recommendation']}</h2>
<b>Phương án tài chính:</b> {decision['selected_financing_option']}<br>
<b>Số tiền:</b> {format_vnd(decision['funding_amount'])}
</div>
\"\"\",
            unsafe_allow_html=True,
        )

        st.markdown("#### Ba lý do")
        for reason in decision["three_reasons"]:
            st.write("• " + reason)

        st.markdown("#### Điều kiện bảo vệ")
        st.warning(decision["protection_condition"])

        risk_level = risk_result["risk_level"]
        if risk_level in {"CRITICAL", "HIGH"}:
            st.error(f"Risk level: {risk_level}")
        elif risk_level == "MEDIUM":
            st.warning(f"Risk level: {risk_level}")
        else:
            st.success(f"Risk level: {risk_level}")

        if result["missing_fields"]:
            st.error("Missing Data Request: " + ", ".join(result["missing_fields"]))

        # BƯỚC BẮT BUỘC, ÁP DỤNG CHO MỌI HỢP ĐỒNG: Founder phải đưa ra quyết định
        # tường minh (Phê duyệt / Từ chối) trước khi Decision Card có hiệu lực. Bước
        # này KHÔNG phụ thuộc vào ngưỡng 300 triệu VND hay mức độ rủi ro — nó tồn tại
        # cho MỌI hợp đồng, không có điều kiện loại trừ nào. Ngưỡng nhạy cảm
        # (requested_amount > 300 triệu / RR-005) chỉ là THÔNG TIN THAM KHẢO thêm để
        # Founder cân nhắc, không quyết định có xuất hiện bước duyệt này hay không.
        sensitive = decision["human_approval_required"] or result["founder_approval_needed"]

        st.markdown("#### Founder Approval Gate")
        st.caption("Mọi hợp đồng đều cần Founder ra quyết định cuối cùng trước khi triển khai.")
        if sensitive:
            st.warning("⚠️ Hợp đồng vượt ngưỡng nhạy cảm (RR-005 / > 300 triệu VND) — cần xem xét kỹ.")
            st.caption(decision["approval_reason"])

        founder_decision = st.radio(
            "Quyết định của Founder",
            options=["Chưa quyết định", "✅ Phê duyệt (Approve)", "❌ Từ chối (Reject)"],
            index=0,
            key="founder_decision_radio",
            horizontal=True,
        )

        if founder_decision == "✅ Phê duyệt (Approve)":
            st.success("APPROVED — Decision Card có hiệu lực, hợp đồng được phép triển khai.")
        elif founder_decision == "❌ Từ chối (Reject)":
            st.error("REJECTED — Founder đã từ chối, hợp đồng không được triển khai.")
        else:
            st.info("PENDING FOUNDER APPROVAL — Chưa có quyết định, đây mới là đề xuất của Agent.")


        with st.expander("Executive Summary"):
            st.write(decision["executive_summary"])"""


new_block = """        # Render Premium Dashboard KPI
        conf_score = f"{decision['confidence_score']:.0%}" if decision["confidence_score"] is not None else "N/A"
        st.markdown(f\"\"\"
        <style>
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .kpi-card {{ background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); border: 1px solid #f1f5f9; transition: transform 0.2s ease; border-top: 4px solid #3b82f6; }}
        .kpi-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
        .kpi-title {{ font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 2.2rem; font-weight: 700; color: #0f172a; line-height: 1.1; }}
        
        .dash-section-title {{ font-size: 1.4rem; font-weight: 700; color: #1e293b; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #f1f5f9; display: flex; align-items: center; gap: 8px; }}
        .reasons-list {{ list-style-type: none; padding: 0; }}
        .reasons-list li {{ background: #f8fafc; margin-bottom: 10px; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #8b5cf6; color: #334155; font-size: 0.95rem; line-height: 1.5; box-shadow: 0 2px 5px rgba(0,0,0,0.01); }}
        </style>
        
        <div class="kpi-grid">
            <div class="kpi-card" style="border-top-color: #3b82f6;">
                <div class="kpi-title">Gross Margin</div>
                <div class="kpi-value">{decision['gross_margin']:.1%}</div>
            </div>
            <div class="kpi-card" style="border-top-color: #10b981;">
                <div class="kpi-title">Closing Cash</div>
                <div class="kpi-value" style="font-size: 1.6rem; padding-top: 5px;">{format_vnd(decision["closing_cash"])}</div>
            </div>
            <div class="kpi-card" style="border-top-color: #f59e0b;">
                <div class="kpi-title">Funding Amount</div>
                <div class="kpi-value" style="font-size: 1.6rem; padding-top: 5px;">{format_vnd(decision['funding_amount'])}</div>
            </div>
            <div class="kpi-card" style="border-top-color: #8b5cf6;">
                <div class="kpi-title">Confidence Score</div>
                <div class="kpi-value">{conf_score}</div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)
        
        dash_col1, dash_col2 = st.columns([1.8, 1.2], gap="large")
        
        with dash_col1:
            # Huge Decision Card
            rec = decision['recommendation']
            rec_color = "#10b981" if "ACCEPT" in rec else "#ef4444" if "REJECT" in rec else "#f59e0b"
            st.markdown(f\"\"\"
            <div style="background: linear-gradient(135deg, #ffffff, #f8fafc); border-radius: 20px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; position: relative; overflow: hidden; margin-bottom: 30px;">
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 6px; background: {rec_color};"></div>
                <div style="font-size: 0.9rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Recommendation</div>
                <h2 style="font-size: 2.8rem; font-weight: 800; color: {rec_color}; margin-top: 0; margin-bottom: 25px; line-height: 1.1;">{rec}</h2>
                
                <div style="display: flex; flex-direction: column; gap: 15px;">
                    <div style="background: #f1f5f9; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0;">
                        <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; margin-bottom: 4px;">SELECTED FINANCING OPTION</div>
                        <div style="font-size: 1.1rem; color: #1e293b; font-weight: 700;">{decision['selected_financing_option']}</div>
                    </div>
                    
                    <div style="background: #fff; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);">
                        <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; margin-bottom: 8px;">EXECUTIVE SUMMARY</div>
                        <div style="font-size: 0.95rem; color: #334155; line-height: 1.6;">{decision['executive_summary']}</div>
                    </div>
                </div>
            </div>
            \"\"\", unsafe_allow_html=True)
            
            st.markdown('<div class="dash-section-title">📌 Lập luận chính (3 Reasons)</div>', unsafe_allow_html=True)
            reasons_html = "<ul class='reasons-list'>" + "".join([f"<li>{r}</li>" for r in decision["three_reasons"]]) + "</ul>"
            st.markdown(reasons_html, unsafe_allow_html=True)

        with dash_col2:
            st.markdown('<div class="dash-section-title">🛡️ Quản trị Rủi ro</div>', unsafe_allow_html=True)
            
            risk_level = risk_result["risk_level"]
            if risk_level in {"CRITICAL", "HIGH"}:
                st.error(f"🚨 **Risk Level: {risk_level}**\\n\\nCần đặc biệt lưu ý và kiểm soát nghiêm ngặt.")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ **Risk Level: {risk_level}**\\n\\nRủi ro có thể chấp nhận nếu tuân thủ điều kiện bảo vệ.")
            else:
                st.success(f"✅ **Risk Level: {risk_level}**\\n\\nHợp đồng ở ngưỡng an toàn.")
                
            st.markdown(f\"\"\"
            <div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px; padding: 16px; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(251, 191, 36, 0.1);">
                <div style="font-size: 0.9rem; color: #b45309; font-weight: 700; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                    <span style="font-size: 1.2rem;">🔒</span> Protection Condition
                </div>
                <div style="font-size: 0.95rem; color: #92400e; line-height: 1.5;">{decision['protection_condition']}</div>
            </div>
            \"\"\", unsafe_allow_html=True)
            
            if result["missing_fields"]:
                st.markdown(f\"\"\"
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px; margin-bottom: 15px;">
                    <div style="font-size: 0.9rem; color: #b91c1c; font-weight: 700; margin-bottom: 8px;">❌ Missing Data Request</div>
                    <div style="font-size: 0.9rem; color: #991b1b;">{', '.join(result['missing_fields'])}</div>
                </div>
                \"\"\", unsafe_allow_html=True)

            st.markdown('<div class="dash-section-title" style="margin-top: 30px;">✍️ Founder Approval Gate</div>', unsafe_allow_html=True)
            
            sensitive = decision["human_approval_required"] or result["founder_approval_needed"]
            if sensitive:
                st.markdown(f\"\"\"
                <div style="background: #fff5f5; border-left: 4px solid #fc8181; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 15px;">
                    <div style="color: #c53030; font-weight: 700; font-size: 0.9rem; margin-bottom: 4px;">⚠️ Hợp đồng vượt ngưỡng nhạy cảm</div>
                    <div style="color: #9b2c2c; font-size: 0.85rem; line-height: 1.4;">{decision['approval_reason']}</div>
                </div>
                \"\"\", unsafe_allow_html=True)
            else:
                st.caption("Mọi hợp đồng đều cần Founder ra quyết định cuối cùng trước khi triển khai.")
                
            founder_decision = st.radio(
                "Quyết định của Founder",
                options=["Chưa quyết định", "✅ Phê duyệt (Approve)", "❌ Từ chối (Reject)"],
                index=0,
                key="founder_decision_radio",
                horizontal=False,
            )

            if founder_decision == "✅ Phê duyệt (Approve)":
                st.success("✅ **APPROVED**\\n\\nDecision Card có hiệu lực, hợp đồng được phép triển khai.")
            elif founder_decision == "❌ Từ chối (Reject)":
                st.error("❌ **REJECTED**\\n\\nFounder đã từ chối, hợp đồng bị hủy bỏ.")
            else:
                st.info("⏳ **PENDING APPROVAL**\\n\\nĐang chờ Founder ra quyết định...")"""


if old_block in content:
    content = content.replace(old_block, new_block)
    with open("g:/OneDrive - Đại học Thương mại/Máy tính/MIS/app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success")
else:
    print("Old block not found!")
