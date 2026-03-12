"""
main.py - FastAPI backend cho hệ thống nhận diện chữ viết tay tiếng Việt.
Sử dụng model CNN + BiLSTM + CTC tự xây dựng.
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File
import uvicorn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_engine.recognizer import OCREngine

app = FastAPI(
    title="API Nhận diện Chữ Viết Tay Tiếng Việt",
    description="Backend API sử dụng model CNN + BiLSTM + CTC tự xây dựng",
    version="2.0.0"
)

# Cấu hình
DEVICE = 'cuda:0' if os.getenv("USE_CUDA", "false").lower() == "true" else 'cpu'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'ocr_model.pth')
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'temp_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Khởi tạo OCR Engine
print("Đang khởi tạo AI Engine...")
ocr_engine = OCREngine(model_path=MODEL_PATH, device=DEVICE)


@app.get("/")
def read_root():
    return {"message": "Hệ thống Backend API OCR đang hoạt động tốt!"}


@app.post("/api/recognize")
async def recognize_handwriting(file: UploadFile = File(...)):
    """Nhận ảnh upload, chạy OCR và trả về text."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        recognized_text = ocr_engine.predict(file_path)
        status = "success"
    except Exception as e:
        recognized_text = f"Đã xảy ra lỗi trong quá trình nhận diện: {str(e)}"
        status = "error"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "filename": file.filename,
        "model_used": "CNN_BiLSTM_CTC",
        "recognized_text": recognized_text,
        "status": status
    }


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)