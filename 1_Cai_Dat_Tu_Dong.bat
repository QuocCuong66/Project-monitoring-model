@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Cài Đặt Tự Động 1-Click
cls

echo ===============================================================================
echo   FALL AND STROKE WARNING SYSTEM — CÀI ĐẶT MÔI TRƯỜNG TỰ ĐỘNG (1-CLICK)
echo ===============================================================================
echo.
echo [*] Đang kiểm tra Python trên máy tính của bạn...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [LỖI] Máy tính của bạn chưa cài đặt Python!
    echo       Vui lòng tải và cài đặt Python từ: https://www.python.org/downloads/
    echo       *LƯU Ý*: Trong lúc cài đặt, hãy tích chọn vào ô:
    echo                [x] Add python.exe to PATH
    echo.
    pause
    exit /b 1
)

echo [*] Đã tìm thấy Python:
python --version

if not exist ".venv" (
    echo.
    echo [*] Đang khởi tạo môi trường ảo siêu nhẹ (.venv)...
    python -m venv .venv
)

echo.
echo [*] Đang cài đặt các thư viện AI siêu nhẹ (ONNX Runtime, FastAPI, OpenCV)...
echo [*] Quá trình này chỉ mất khoảng 1 - 2 phút tùy theo tốc độ mạng...
echo.

.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ===============================================================================
echo  [THÀNH CÔNG] Đã hoàn tất cài đặt toàn bộ môi trường!
echo.
echo  Từ bây giờ, bạn chỉ cần nhấp đúp vào file:
echo  👉 2_Chay_He_Thong.bat (hoặc Run_Elderly_Monitor.bat) để sử dụng!
echo ===============================================================================
echo.
pause
