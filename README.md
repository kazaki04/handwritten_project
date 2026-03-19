# DỰ ÁN NHẬN DẠNG CHỮ VIẾT TAY TIẾNG VIỆT

## Kiến trúc hệ thống
Hệ thống của chúng tôi được thiết kế theo dạng module hóa, giúp tập thể dễ dàng bảo trì và mở rộng sau này, bao gồm các thành phần chính:
- Giao diện người dùng (Frontend): Phát triển bằng Streamlit, giúp người dùng cuối dễ dàng tương tác và tải hình ảnh lên hệ thống.
- Máy chủ xử lý (Backend): Xây dựng trên nền tảng FastAPI, đóng vai trò nền tảng điều phối luồng dữ liệu an toàn từ giao diện tới các mô hình AI trực tiếp trong bộ nhớ.
- Hệ thống Trích xuất Vùng Text (Detection): Sử dụng bộ định vị PaddleOCR để nhận diện, phân tích và khoanh vùng các đoạn văn bản xuất hiện trong không gian ảnh.
- Hệ thống Nhận dạng (Recognition): Tích hợp lõi VietOCR kết hợp với kiến trúc vgg_transformer để đọc hiểu và chuyển đổi các hình ảnh đã khoanh vùng thành chuỗi ký tự Tiếng Việt chuẩn.
- Mô-đun Hậu xử lý (Post-Processing): Hệ thống xử lý ngôn ngữ quy tắc và máy học, hỗ trợ chuẩn hóa cấu trúc từ vựng và tự động sửa các lỗi chính tả phát sinh sau nhận dạng.

## Yêu cầu hệ thống
Để đảm bảo tính đồng bộ, ổn định và tránh lỗi phát sinh trong môi trường Windows, nhóm phát triển đã thiết lập môi trường với các thông số kỹ thuật nội bộ cụ thể:
- Python 3.9 (hoặc các phiên bản tương thích cấu trúc).
- Các thư viện cốt lõi tuân thủ chặt chẽ phiên bản kiểm thử: fastapi, streamlit, opencv-python, paddlepaddle (khóa cứng ở bản 2.6.2), paddleocr (khóa cứng ở bản 2.9.1), vietocr (khóa cứng ở bản 0.3.13) và numpy.

## Hướng dẫn cài đặt và sử dụng

1. Thiết lập môi trường phát triển:
Chúng tôi khuyến nghị tất cả thành viên và người cài đặt mới sử dụng môi trường ảo (virtual environment) để thiết lập các thư viện nhằm tránh mọi rủi ro xung đột phần mềm hệ thống:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Khởi chạy hệ thống:
Nhóm đã đồng thuận tích hợp sẵn một tập lệnh điều phối duy nhất (PowerShell script) để việc khởi chạy cả hệ thống được đồng bộ thao tác. Người dùng chỉ cần mở Terminal tại thư mục gốc của dự án và khởi chạy lệnh sau:
```bash
.\run_all.ps1 -Action start
```
Lệnh này sẽ rà soát các cổng dịch vụ cũ đang tổn đọng, tự động thu hồi và tái khởi động an toàn cho cả Backend API (cổng mặc định 8000) cũng như Frontend UI (cổng mặc định 8501).

3. Trải nghiệm hệ thống:
Mở trình duyệt web và truy cập vào địa chỉ nội bộ: http://localhost:8501 để bắt đầu tương tác với giao diện nhận diện.

4. Tạm dừng hệ thống:
Khi tập thể hoàn tất công việc hoặc người dùng muốn giải phóng tài nguyên máy, toàn bộ tiến trình liên quan có thể được tắt một cách an toàn bằng lệnh:
```bash
.\run_all.ps1 -Action stop
```

## Cấu trúc thư mục dự án
Dự án được tổ chức một cách chặt chẽ, tạo điều kiện thuật tiện cho việc phân chia công việc trong tập thể nhóm:
- data/: Chứa tài nguyên số liệu, dữ liệu huấn luyện và kiểm thử liên quan.
- logs/: Nơi lưu trữ nhất thống các tập tin thông báo quá trình (log) từ backend và frontend.
- models/: Thư mục lưu trữ các tệp trọng số đã qua quá trình huấn luyện (ví dụ: vietocr_model.pth).
- src/ai_engine/: Chứa toàn bộ các tệp mã nguồn logic lõi Trí tuệ nhân tạo (detection.py, recognizer.py, post_processing.py).
- src/api/: Cấu trúc luồng và phân tuyến (router) của máy chủ FastAPI trung gian.
- src/ui/: Mã nguồn thiết kế màn hình giao diện hiển thị Streamlit.
- run_all.ps1: Tập lệnh vỏ (shell) tự động giúp quản lý luồng khởi động đa dịch vụ.
- requirements.txt: Liệt kê chặt chẽ danh sách các thư viện phụ thuộc của môi trường dự án.
