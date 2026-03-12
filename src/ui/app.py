import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/api/recognize"

st.set_page_config(page_title="AI Nhận diện chữ viết tay", layout="wide", page_icon="📝")

st.title("Hệ thống Nhận diện Chữ viết tay Tiếng Việt")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Cài đặt & Tải ảnh")

    model_options = {
        "VietOCR Mặc định (Tốc độ nhanh)": "vietocr_base",
        "Mô hình Tự Train - ResNet (Độ chính xác cao)": "custom_resnet"
    }
    selected_model_label = st.selectbox("Chọn cấu hình AI Engine:", list(model_options.keys()))
    selected_model_key = model_options[selected_model_label]
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Tải ảnh chứa chữ viết tay lên đây:", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh gốc", use_column_width=True)

with col2:
    st.header("2. Kết quả nhận diện")
    
    if uploaded_file is None:
        st.warning("Vui lòng tải ảnh lên ở cột bên trái để bắt đầu.")
    else:
        with st.form("ocr_form"):
            submit_button = st.form_submit_button("Chạy Mô Hình Nhận Diện", use_container_width=True)
            
        if submit_button:
            with st.spinner(f'Đang trích xuất văn bản bằng {selected_model_label}...'):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"model_type": selected_model_key}

                    response = requests.post(API_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        result_text = res_data.get("recognized_text", "")
                        
                        st.success("Hoàn tất trích xuất!")
                        edited_text = st.text_area(
                            "Văn bản nhận diện:", 
                            value=result_text, 
                            height=250
                        )
                        
                        st.download_button(
                            label="Tải xuống File Text (.txt)",
                            data=edited_text,
                            file_name=f"ket_qua.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Lỗi Server: Mã {response.status_code}")
                except Exception as e:
                    st.error(f"Không thể kết nối với Backend API. Lỗi: {e}")