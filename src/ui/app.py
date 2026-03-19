"""
app.py - Giao diện Streamlit cho hệ thống nhận diện chữ viết tay tiếng Việt.
"""

import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/api/recognize"
MODEL_INFO_URL = "http://localhost:8000/api/model-info"

st.set_page_config(page_title="AI Nhận diện chữ viết tay", layout="wide", page_icon="📝")

st.title("Hệ thống Nhận diện Chữ viết tay Tiếng Việt")
st.markdown("**Model: CNN + BiLSTM + CTC (tự xây dựng)**")

try:
    model_info_resp = requests.get(MODEL_INFO_URL, timeout=10)
    if model_info_resp.status_code == 200:
        model_info = model_info_resp.json()
        if model_info.get("loaded"):
            st.caption(
                f"Best model: `{model_info.get('model_name', 'ocr_model.pth')}` | "
                f"Epoch: {model_info.get('epoch', 'N/A')} | "
                f"Val CER: {model_info.get('val_cer', 'N/A')}"
            )
        else:
            st.caption("Chưa load được trọng số model, backend đang dùng random weights.")
except Exception:
    st.caption("Không lấy được thông tin model từ backend.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Tải ảnh lên")
    uploaded_file = st.file_uploader(
        "Tải ảnh chứa chữ viết tay lên đây:", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh gốc", use_container_width=True)

with col2:
    st.header("2. Kết quả nhận diện")

    if uploaded_file is None:
        st.warning("Vui lòng tải ảnh lên ở cột bên trái để bắt đầu.")
    else:
        with st.form("ocr_form"):
            submit_button = st.form_submit_button(
                "Chạy Mô Hình Nhận Diện", use_container_width=True
            )

        if submit_button:
            with st.spinner("Đang trích xuất văn bản..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    response = requests.post(API_URL, files=files, timeout=60)

                    if response.status_code == 200:
                        res_data = response.json()
                        result_text = res_data.get("recognized_text", "")

                        st.success("Hoàn tất trích xuất!")
                        edited_text = st.text_area(
                            "Văn bản nhận diện:", value=result_text, height=250
                        )

                        st.download_button(
                            label="Tải xuống File Text (.txt)",
                            data=edited_text,
                            file_name="ket_qua.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"Lỗi Server: Mã {response.status_code}")
                except Exception as e:
                    st.error(f"Không thể kết nối với Backend API. Lỗi: {e}")