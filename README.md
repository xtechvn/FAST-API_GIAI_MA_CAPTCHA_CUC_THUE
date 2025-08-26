# Hướng dẫn sử dụng Captcha Solver API
tao file de luon chay
sudo nano /etc/systemd/system/fastapi-captcha.service

source /var/www/venv/bin/activate

sudo nano /etc/nginx/sites-available/captcha
server {
    listen 80;
    server_name solver-captcha.adavigo.com;

   server {
    listen 80;
    server_name solver-captcha.adavigo.com;

    location / {
        proxy_pass https://solver-captcha.adavigo.com;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

Bước 4: Run ứng dụng với Uvicorn + Systemd

Tạo service để chạy FastAPI như daemon:

sudo nano /etc/systemd/system/captcha.service

Nội dung:

[Unit]
Description=FastAPI CAPTCHA Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/FAST-API_GIAI_MA_CAPTCHA_CUC_THUE
ExecStart=/var/www/FAST-API_GIAI_MA_CAPTCHA_CUC_THUE/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

}



Dự án này cung cấp một API sử dụng FastAPI và Deep Learning để tự động giải mã captcha.

---

## Bước 1: Cài đặt môi trường

1. Đảm bảo máy bạn đã cài đặt Python 3.8 trở lên.
2. Cài đặt các thư viện cần thiết bằng lệnh:
   ```
   pip install -r requirements.txt
   ```
3. (Tùy chọn) Tạo môi trường ảo để quản lý thư viện:
   ```
   python -m venv venv
   source venv/bin/activate  # Trên Linux/Mac
   venv\Scripts\activate     # Trên Windows
   ```

## Bước 2: Khởi động server

Chạy lệnh sau để khởi động API:


monitor
#!/bin/bash

# ===============================
# Cấu hình Telegram
# ===============================
TOKEN="8448060409:AAFqpFVAirNaosuBz7Shy74XXqCXSyrOT5c"
CHAT_ID="-1002893753557"  # thay bằng chat_id của bạn

# Ngưỡng cảnh báo
RAM_THRESHOLD=80       # % RAM sử dụng
SWAP_THRESHOLD=50      # % swap sử dụng
DISK_THRESHOLD=80      # % ổ cứng sử dụng
CPU_THRESHOLD=90       # % CPU tiến trình

# Lấy thông tin RAM
RAM_USED=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
SWAP_USED=$(free | awk '/Swap:/ {printf "%.0f", $3/$2 * 100}')

# Lấy thông tin ổ cứng root
DISK_USED=$(df / | awk 'NR==2 {print $5}' | tr -d '%')

# Lấy top tiến trình chiếm CPU cao nhất
TOP_CPU=$(ps -eo pid,cmd,%cpu --sort=-%cpu | head -n 2 | tail -n 1)
TOP_CPU_PERC=$(echo $TOP_CPU | awk '{print $3}' | cut -d. -f1)

# Tạo thông báo nếu vượt ngưỡng
MESSAGE=""

if [ $RAM_USED -ge $RAM_THRESHOLD ]; then
    MESSAGE+="⚠️ RAM đang sử dụng $RAM_USED% (ngưỡng $RAM_THRESHOLD%)\n"
fi

if [ $SWAP_USED -ge $SWAP_THRESHOLD ]; then
    MESSAGE+="⚠️ Swap đang sử dụng $SWAP_USED% (ngưỡng $SWAP_THRESHOLD%)\n"
fi

if [ $DISK_USED -ge $DISK_THRESHOLD ]; then
    MESSAGE+="⚠️ Ổ cứng / đang sử dụng $DISK_USED% (ngưỡng $DISK_THRESHOLD%)\n"
fi

if [ $TOP_CPU_PERC -ge $CPU_THRESHOLD ]; then
    PROC_NAME=$(echo $TOP_CPU | awk '{for(i=2;i<=NF-1;i++) printf $i " ";}')
    MESSAGE+="⚠️ Tiến trình $PROC_NAME đang dùng CPU $TOP_CPU_PERC% (ngưỡng $CPU_THRESHOLD%)\n"
fi

# Gửi cảnh báo nếu có
if [ ! -z "$MESSAGE" ]; then
    curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT_ID" \
    -d text="$MESSAGE"
fi
