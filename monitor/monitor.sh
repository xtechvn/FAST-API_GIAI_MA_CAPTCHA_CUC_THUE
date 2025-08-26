#!/bin/bash

# ===============================
# Cấu hình Telegram
# ===============================
TOKEN="8448060409:AAFqpFVAirNaosuBz7Shy74XXqCXSyrOT5c"
CHAT_ID="-1002893753557"  # Bot Monitor Linux 106.163.216.33

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