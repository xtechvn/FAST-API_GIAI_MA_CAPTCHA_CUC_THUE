# Captcha Solver API

Ứng dụng FastAPI để giải mã captcha sử dụng Deep Learning.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:
```bash
python main.py
```

hoặc

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Kiểm tra trạng thái
- **GET** `/health`
- Trả về trạng thái của API và model

### 2. Giải mã captcha từ file
- **POST** `/solve-captcha`
- Upload file ảnh captcha để giải mã
- Form data: `file` (image file)

### 3. Giải mã captcha từ base64
- **POST** `/solve-captcha-base64`
- Gửi ảnh dưới dạng base64 string
- JSON: `{"image": "base64_string"}`

### 4. Trạng thái model
- **GET** `/model-status`
- Kiểm tra trạng thái load model

## Cấu trúc dự án

```
PY-GIAI_MA_CAPTCHA/
├── main.py              # File chính FastAPI
├── requirements.txt     # Danh sách thư viện
├── models/             # Thư mục chứa model đã train (tùy chọn)
└── README.md           # Hướng dẫn sử dụng
```

## Lưu ý

- Model hiện tại chưa được load (cần train model trước)
- Cần thay thế đường dẫn model trong `load_model()` method
- API hiện tại trả về placeholder "CAPTCHA_RESULT"

## Development

Để phát triển thêm:

1. Train model deep learning cho captcha
2. Cập nhật `load_model()` method để load model thực
3. Implement `decode_prediction()` method để xử lý output của model
4. Thêm preprocessing tùy theo loại captcha cụ thể

## API Documentation

Sau khi chạy server, truy cập:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc