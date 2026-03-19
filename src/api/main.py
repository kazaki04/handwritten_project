import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
from PIL import Image
import sys

# Thêm root vào sys.path để import các module local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_engine.detection import TextDetector
from src.ai_engine.recognizer import OCREngine
from src.ai_engine.post_processing import SpellCorrector

app = FastAPI(
    title="Vietnamese Handwritten Text Recognition API",
    description="Backend API using PaddleOCR for detection and VietOCR for recognition.",
    version="3.0.0"
)

# --- Configuration ---
DEVICE = 'cpu' 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Đường dẫn file trọng số VietOCR
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vietocr_model.pth')

# --- Initialize AI Engines ---
print("Initializing AI Engines (CPU Mode)...")
# PaddleOCR detector
text_detector = TextDetector(lang='vi') 
# VietOCR engine
ocr_engine = OCREngine(model_path=MODEL_PATH, device=DEVICE)
# Spell Corrector
spell_corrector = SpellCorrector()

def _model_meta():
    info = ocr_engine.model_info
    return {
        "loaded": info.get("loaded", False),
        "model_path": info.get("model_path"),
        "model_name": os.path.basename(info.get("model_path") or ""),
        "device": DEVICE,
    }

@app.get("/")
def read_root():
    return {"message": "Handwritten OCR Backend API is running on CPU!"}

@app.get("/api/model-info")
def model_info():
    """Returns information about the loaded recognition model."""
    return _model_meta()

@app.post("/api/recognize")
async def recognize_handwriting(file: UploadFile = File(...)):
    """
    Nhận diện chữ viết tay: Phát hiện vùng chữ -> Nhận diện từng vùng -> Ghép văn bản.
    """
    full_text = []
    status = "success"
    
    try:
        # 1. Đọc file từ UploadStream trực tiếp vào bộ nhớ (Memory)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Không thể giải mã hình ảnh. Vui lòng kiểm tra định dạng file.")

        # 2. Text Detection (Sử dụng PaddleOCR)
        # cls=False để tránh lỗi tham số, det=True để lấy bounding boxes
        detection_results = text_detector.ocr.ocr(img, cls=False, det=True, rec=False)

        if not detection_results or detection_results[0] is None:
            return {
                "filename": file.filename,
                "recognized_text": "Không tìm thấy vùng văn bản nào trong ảnh.",
                "status": "success"
            }

        # Sắp xếp các box từ trên xuống dưới để văn bản đọc tự nhiên hơn
        boxes = detection_results[0]
        # (Tùy chọn) Sắp xếp theo tọa độ y của điểm đầu tiên trong box
        boxes.sort(key=lambda x: x[0][1])

        for box in boxes:
            # Xác định tọa độ bao quanh vùng chữ
            pts = np.array(box, dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            
            # Cắt ảnh vùng chữ (crop) - Đảm bảo tọa độ không âm
            y1, y2 = max(0, int(y_min)), int(y_max)
            x1, x2 = max(0, int(x_min)), int(x_max)
            cropped_img_np = img[y1:y2, x1:x2]
            
            if cropped_img_np.size == 0:
                continue
                
            # Chuyển đổi sang PIL Image để tương thích với VietOCR
            cropped_img_rgb = cv2.cvtColor(cropped_img_np, cv2.COLOR_BGR2RGB)
            cropped_img_pil = Image.fromarray(cropped_img_rgb)

            # 3. Text Recognition (Sử dụng VietOCR)
            # Quan trọng: Đảm bảo ocr_engine.predict() nhận vào một PIL Image
            recognized_line = ocr_engine.predict(cropped_img_pil)
            
            if recognized_line:
                full_text.append(recognized_line.strip())

        # 4. Hậu xử lý văn bản
        final_text = " ".join(full_text)
        corrected_text = spell_corrector.correct(final_text)
        
    except Exception as e:
        corrected_text = f"An error occurred during recognition: {str(e)}"
        status = "error"

    return {
        "filename": file.filename,
        "model_used": _model_meta(),
        "recognized_text": corrected_text,
        "status": status
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)