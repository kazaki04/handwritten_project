import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

from src.ai_engine.recognizer import VietOCREngine

app = FastAPI(
    title="API Nhận diện Chữ Viết Tay Tiếng Việt",
    description="Backend API xử lý ảnh với tuỳ chọn Multi-Model",
    version="2.0.0"
)

DEVICE = 'cuda:0' if os.getenv("USE_CUDA", "true").lower() == "true" else 'cpu'

print("Đang khởi tạo AI Engine...")
ocr_engine = VietOCREngine(device=DEVICE)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Hệ thống Backend API đang hoạt động tốt!"}

@app.post("/api/recognize")
async def recognize_handwriting(
    file: UploadFile = File(...),
    model_type: str = Form("vietocr_base") 
):

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        recognized_text = ocr_engine.predict(file_path, model_type=model_type)
        status = "success"
    except Exception as e:
        recognized_text = f"Đã xảy ra lỗi trong quá trình nhận diện: {str(e)}"
        status = "error"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return {
        "filename": file.filename,
        "model_used": model_type,
        "recognized_text": recognized_text,
        "status": status
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)