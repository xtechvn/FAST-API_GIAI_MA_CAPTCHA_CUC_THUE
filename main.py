from fastapi import FastAPI, HTTPException
import numpy as np
from PIL import Image
import io
import base64
import logging
import tensorflow as tf
from pydantic import BaseModel
from predict_captcha.PrecisionCaptchaCNN import predict_captcha_base64
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Captcha Solver API",
    description="API để dịch và giải mã captcha sử dụng Deep Learning",
    version="1.0.0"
)


class CaptchaRequest(BaseModel):
    image: str  # base64 string


class CaptchaSolver:
    def __init__(self):
        self.model = None
        self.is_model_loaded = False

    def load_model(self):
        """Load model deep learning đã được train"""
        try:
            # TODO: Thay thế bằng đường dẫn model thực tế
            self.model = tf.keras.models.load_model('model/precision_model_best_no_grid.h5')
            logger.info("Model loaded successfully")
            self.is_model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_model_loaded = False


# Khởi tạo solver
captcha_solver = CaptchaSolver()


@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi app start"""
    logger.info("Starting Captcha Solver API...")
    # Load model khi khởi động
    captcha_solver.load_model()


@app.get("/")
async def root():
    """Endpoint gốc"""
    return {
        "message": "Captcha Solver API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "model_loaded": captcha_solver.is_model_loaded
    }


@app.post("/solve-captcha-base64")
async def solve_captcha_base64(request: CaptchaRequest):
    """
    Gọi endpoint này bằng cách gửi POST request tới:
    http://localhost:8000/solve-captcha-base64

    Body (JSON):
        {
            "image": "<base64_string>"
        }
    """
    if not captcha_solver.is_model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_base64 = request.image
    # print(image_base64)
    try:
        import time
        # Kiểm tra input hợp lệ
        if not image_base64 or not isinstance(image_base64, str):
            raise ValueError("Image data is missing or invalid")

        start_time = time.time()
        # predict_captcha_base64 có thể raise lỗi nếu base64 không hợp lệ
        result = predict_captcha_base64(image_base64)
        elapsed_time = time.time() - start_time

        if result is None or result == "":
            raise ValueError("Could not solve captcha or empty result")
        return {
            "result": result,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error solving captcha: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.get("/model-status")
async def get_model_status():
    """Lấy trạng thái của model"""
    return {
        "model_loaded": captcha_solver.is_model_loaded,
        "model_info": "Captcha Solver Model v1.0"
    }
# End of Selection