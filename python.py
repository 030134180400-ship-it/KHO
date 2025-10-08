# app_pt_duan.py

import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import json
import io

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Phương Án Kinh Doanh 💰")

# --- Hàm tính toán chính ---

def calculate_project_metrics(V0, N, R_annual, C_annual, tax_rate, WACC):
    """
    Tính toán Bảng Dòng tiền, NPV, IRR, PP, DPP.
    V0: Vốn đầu tư ban đầu (Vốn đầu tư).
    N: Vòng đời dự án (năm).
    R_annual: Doanh thu hàng năm.
    C_annual: Chi phí hàng năm.
    tax_rate: Thuế suất (%).
    WACC: Chi phí vốn bình quân (%).
    """
    
    # 1. Tính Dòng tiền (Cash Flow - CF)
    # Lợi nhuận trước thuế (EBT) = Doanh thu - Chi phí
    EBT = R_annual - C_annual
    
    # Lợi nhuận sau thuế (EAT)
    EAT = EBT * (1 - tax_rate)
    
    # Dòng tiền thuần (CF) = EAT + Khấu hao (Giả định Khấu hao = 0 trong trường hợp này 
    # vì dữ liệu thô không có, chỉ có EBT/EAT từ (Doanh thu - Chi phí))
    # Trong thực tế: CF = EAT + Khấu hao. Ở đây, CF = EAT.
    CF_annual = EAT

    # Khởi tạo bảng dòng tiền
    years = list(range(0, N + 1))
    
    # Dòng tiền tại Năm 0 là Vốn đầu tư ban đầu (âm)
    cash_flows = [-V0] + [CF_annual] * N
    
    # 2. Xây dựng DataFrame Dòng tiền
    df_cf = pd.DataFrame({
        'Năm': years,
        'Dòng tiền ($CF_t$)': cash_flows,
        'Hệ số chiết khấu ($1/(1+WACC)^t$)': [1 / ((1 + WACC)**t) for t in years]
    })
    
    # Dòng tiền chiết khấu ($DCF_t$)
    df_cf['Dòng tiền chiết khấu ($DCF_t$)'] = df_cf['Dòng tiền ($CF_t$)'] * df_cf['Hệ số chiết khấu ($1/(1+WACC)^t$)']
    
    # Dòng tiền tích lũy chiết khấu ($CDCF_t$)
    df_cf['Dòng tiền tích lũy chiết khấu ($CDCF_t$)'] = df_cf['Dòng tiền chiết khấu ($DCF_t$)'].cumsum()
    
    # Dòng tiền tích lũy không chiết khấu ($CCF_t$)
    df_cf['Dòng tiền tích lũy ($CCF_t$)'] = df_cf['Dòng tiền ($CF_t$)'].cumsum()
    
    # 3. Tính NPV
    NPV = df_cf['Dòng tiền chiết khấu ($DCF_t$)'].sum()
    
    # 4. Tính IRR
    try:
        IRR = np.irr(cash_flows)
    except Exception:
        IRR = np.nan
        
    # 5. Tính PP (Payback Period - Thời gian hoàn vốn)
    df_cf_pos = df_cf[df_cf['Năm'] >= 1]
    
    # Tìm năm cuối cùng mà CCF < 0
    last_neg_year = df_cf[df_cf['Dòng tiền tích lũy ($CCF_t$)'] < 0]['Năm'].max()
    if pd.isna(last_neg_year) or last_neg_year == 0:
        PP = abs(df_cf.loc[0, 'Dòng tiền tích lũy ($CCF_t$)']) / CF_annual
    else:
        # Năm PP = Năm cuối CCF âm + (Giá trị âm cuối cùng / Dòng tiền năm kế tiếp)
        PP_residual = abs(df_cf.loc[last_neg_year, 'Dòng tiền tích lũy ($CCF_t$)'])
        PP = last_neg_year + (PP_residual / CF_annual)
        
    # 6. Tính DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    # Tìm năm cuối cùng mà CDCF < 0
    last_neg_year_d = df_cf[df_cf['Dòng tiền tích lũy chiết khấu ($CDCF_t$)'] < 0]['Năm'].max()
    
    if pd.isna(last_neg_year_d) or last_neg_year_d == 0:
        # Nếu đã hoàn vốn ngay năm đầu tiên (rất hiếm)
        DPP_residual = abs(df_cf.loc[0, 'Dòng tiền tích lũy chiết khấu ($CDCF_t$)'])
        DCF_next = df_cf.loc[1, 'Dòng tiền chiết khấu ($DCF_t$)']
        DPP = DPP_residual / DCF_next if DCF_next != 0 else np.nan
    else:
        # Năm DPP = Năm cuối CDCF âm + (Giá trị âm cuối cùng / Dòng tiền chiết khấu năm kế tiếp)
        DPP_residual = abs(df_cf.loc[last_neg_year_d, 'Dòng tiền tích lũy chiết khấu ($CDCF_t$)'])
        DCF_next = df_cf.loc[last_neg_year_d + 1, 'Dòng tiền chiết khấu ($DCF_t$)']
        DPP = last_neg_year_d + (DPP_residual / DCF_next) if DCF_next != 0 else np.nan

    return df_cf, NPV, IRR, PP, DPP

# --- Hàm gọi API Gemini để trích xuất dữ liệu (Output JSON) ---

def extract_financial_data(project_text, api_key):
    """Sử dụng Gemini để trích xuất các thông số tài chính thành JSON."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Định dạng yêu cầu output JSON
        json_schema = {
            "type": "object",
            "properties": {
                "Vốn đầu tư ban đầu (tỷ VNĐ)": {"type": "number"},
                "Vòng đời dự án (năm)": {"type": "integer"},
                "Doanh thu hàng năm (tỷ VNĐ)": {"type": "number"},
                "Chi phí hàng năm (tỷ VNĐ)": {"type": "number"},
                "Thuế suất (%)": {"type": "number"},
                "WACC (%)": {"type": "number"}
            },
            "required": [
                "Vốn đầu tư ban đầu (tỷ VNĐ)",
                "Vòng đời dự án (năm)",
                "Doanh thu hàng năm (tỷ VNĐ)",
                "Chi phí hàng năm (tỷ VNĐ)",
                "Thuế suất (%)",
                "WACC (%)"
            ]
        }
        
        prompt = f"""
        Bạn là chuyên gia trích xuất dữ liệu tài chính. Hãy đọc văn bản về phương án kinh doanh dưới đây và trích xuất các thông số chính vào định dạng JSON.
        *LƯU Ý QUAN TRỌNG: Tất cả các giá trị phải được chuyển đổi về đơn vị cơ bản TỶ. Ví dụ: 30 tỷ thành 30.
        *Vốn đầu tư: 30 tỷ -> 30
        *Doanh thu: 3.5 tỷ -> 3.5
        *Chi phí: 2 tỷ -> 2
        *Thuế suất: 20% -> 0.20
        *WACC: 13% -> 0.13
        
        Văn bản phương án kinh doanh:
        ---
        {project_text}
        ---
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": json_schema}
        )
        
        # Parse JSON
        data = json.loads(response.text)
        
        # Chuyển đổi lại đơn vị để tính toán (tỷ VNĐ thành VNĐ gốc)
        data_processed = {
            'Vốn đầu tư ban đầu': data['Vốn đầu tư ban đầu (tỷ VNĐ)'] * 1_000_000_000,
            'Vòng đời dự án': data['Vòng đời dự án (năm)'],
            'Doanh thu hàng năm': data['Doanh thu hàng năm (tỷ VNĐ)'] * 1_000_000_000,
            'Chi phí hàng năm': data['Chi phí hàng năm (tỷ VNĐ)'] * 1_000_000_000,
            'Thuế suất': data['Thuế suất (%)'],
            'WACC': data['WACC (%)']
        }
        return data_processed, None

    except APIError as e:
        return None, f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return None, f"Lỗi trích xuất dữ liệu: {e}. Vui lòng kiểm tra định dạng văn bản."

# --- Hàm gọi API Gemini để phân tích hiệu quả dự án ---

def get_ai_analysis_metrics(metrics_data, df_cf, api_key):
    """Sử dụng Gemini để phân tích các chỉ số hiệu quả dự án."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        metrics_df = pd.DataFrame([metrics_data])
        
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Hãy đọc các chỉ số hiệu quả dự án sau và đưa ra một đánh giá chuyên sâu, khách quan, và hành động đề xuất.
        
        1. **Đánh giá tổng thể**: Nhận xét về tính khả thi tài chính của dự án dựa trên NPV và so sánh IRR với WACC (13%).
        2. **Đánh giá Rủi ro**: Nhận xét về thời gian hoàn vốn (PP và DPP).
        3. **Khuyến nghị**: Dự án có nên được triển khai hay không?
        
        Dữ liệu Tóm tắt Chỉ số:
        {metrics_df.to_markdown(index=False)}
        
        Bảng Dòng tiền chi tiết:
        {df_cf.to_markdown(index=False)}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi phân tích: {e}"

# --- Giao diện Streamlit ---

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    st.stop()

st.info("💡 **Hướng dẫn:** Vui lòng dán toàn bộ nội dung phương án kinh doanh (vốn đầu tư, dòng đời, doanh thu, chi phí, WACC, thuế) vào ô bên dưới.")

# 1. Nhập liệu
project_input = st.text_area(
    "1. Dán nội dung Phương án Kinh doanh từ file Word vào đây:",
    height=300,
    value="VỐN ĐẦU TƯ: 30 TỶ VNĐ. Vòng đời dự án: 10 năm. Doanh thu hàng năm: 3.5 TỶ VNĐ. Chi phí hàng năm: 2 TỶ VNĐ. Thuế suất: 20%. WACC: 13%."
)

if st.button("LỌC DỮ LIỆU & PHÂN TÍCH HIỆU QUẢ DỰ ÁN"):
    if not project_input.strip():
        st.warning("Vui lòng dán nội dung phương án kinh doanh.")
    else:
        with st.spinner('Đang dùng AI để trích xuất dữ liệu...'):
            extracted_data, error = extract_financial_data(project_input, api_key)

        if error:
            st.error(error)
            st.stop()

        if extracted_data:
            st.success("✅ Trích xuất dữ liệu thành công!")

            V0 = extracted_data['Vốn đầu tư ban đầu']
            N = extracted_data['Vòng đời dự án']
            R_annual = extracted_data['Doanh thu hàng năm']
            C_annual = extracted_data['Chi phí hàng năm']
            tax_rate = extracted_data['Thuế suất']
            WACC = extracted_data['WACC']

            # Hiển thị dữ liệu đã trích xuất
            st.subheader("2. Dữ liệu đã trích xuất từ AI")
            st.dataframe(
                pd.DataFrame(extracted_data, index=["Giá trị"]).T.style.format({
                    'Giá trị': lambda x: f"{x:,.0f}" if x >= 1000 or x <= -1000 else f"{x:.2f}"
                }),
                use_container_width=True
            )
            
            # --- Tính toán các chỉ số ---
            with st.spinner('Đang tính toán Bảng Dòng tiền và các chỉ số...'):
                try:
                    df_cf, NPV, IRR, PP, DPP = calculate_project_metrics(V0, N, R_annual, C_annual, tax_rate, WACC)
                    
                    # 3. Bảng Dòng tiền
                    st.subheader("3. Bảng Dòng tiền và Tích lũy (Vòng đời 10 năm)")
                    st.dataframe(df_cf.style.format({
                        'Dòng tiền ($CF_t$)': '{:,.0f}',
                        'Hệ số chiết khấu ($1/(1+WACC)^t$)': '{:.4f}',
                        'Dòng tiền chiết khấu ($DCF_t$)': '{:,.0f}',
                        'Dòng tiền tích lũy chiết khấu ($CDCF_t$)': '{:,.0f}',
                        'Dòng tiền tích lũy ($CCF_t$)': '{:,.0f}',
                    }), use_container_width=True)
                    
                    # 4. Chỉ số Đánh giá Hiệu quả
                    st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
                    
                    metrics_data = {
                        'Chỉ số': ['NPV (Giá trị hiện tại ròng)', 'IRR (Tỷ suất sinh lợi nội tại)', 'PP (Thời gian hoàn vốn)', 'DPP (Thời gian hoàn vốn có chiết khấu)'],
                        'Giá trị': [NPV, IRR, PP, DPP],
                        'Đơn vị': ['VNĐ', '%', 'Năm', 'Năm']
                    }
                    df_metrics = pd.DataFrame(metrics_data)

                    # Định dạng hiển thị
                    df_metrics_display = df_metrics.copy()
                    df_metrics_display.loc[0, 'Giá trị'] = f"{NPV:,.0f} VNĐ"
                    df_metrics_display.loc[1, 'Giá trị'] = f"{IRR * 100:.2f} %" if not np.isnan(IRR) else "N/A"
                    df_metrics_display.loc[2, 'Giá trị'] = f"{PP:.2f} Năm" if not np.isnan(PP) else "N/A"
                    df_metrics_display.loc[3, 'Giá trị'] = f"{DPP:.2f} Năm" if not np.isnan(DPP) else "N/A"
                    
                    df_metrics_display = df_metrics_display.drop(columns=['Đơn vị'])
                    st.dataframe(df_metrics_display, hide_index=True, use_container_width=True)

                    # 5. Phân tích AI
                    st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                    
                    metrics_for_ai = {
                        'WACC': f"{WACC * 100:.2f}%",
                        'NPV (VNĐ)': NPV,
                        'IRR (%)': IRR * 100,
                        'PP (Năm)': PP,
                        'DPP (Năm)': DPP
                    }
                    
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis_metrics(metrics_for_ai, df_cf, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)

                except Exception as e:
                    st.error(f"Lỗi tính toán: Không thể tính toán các chỉ số. Vui lòng kiểm tra lại dữ liệu trích xuất. Chi tiết lỗi: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Ứng dụng được xây dựng bởi Chuyên gia Lập trình Python & Streamlit.")
