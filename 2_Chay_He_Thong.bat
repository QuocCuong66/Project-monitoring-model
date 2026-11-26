@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Hệ Thống Cảnh Báo Té Ngã & Đột Quỵ
cls

:: Kiểm tra nếu chưa cài đặt môi trường ảo .venv thì tự động chạy cài đặt
if not exist ".venv\Scripts\python.exe" (
    echo [*] Phát hiện hệ thống chưa được cài đặt lần đầu.
    echo [*] Đang tự động chuyển sang trình cài đặt 1-Click...
    echo.
    call "1_Cai_Dat_Tu_Dong.bat"
    if not exist ".venv\Scripts\python.exe" (
        echo [Lỗi] Quá trình cài đặt chưa thành công. Không thể khởi động.
        pause
        exit /b 1
    )
    cls
)

echo ===============================================================================
echo      FALL AND STROKE WARNING SYSTEM — HỆ THỐNG CẢNH BÁO AN TOÀN
echo ===============================================================================
echo  [*] Trạng thái: Đang khởi động AI Engine và Web Dashboard...
echo  [*] Tăng tốc phần cứng: Tự động nhận diện GPU / DirectML / CPU
echo  [*] Địa chỉ Web Dashboard: http://localhost:8000
echo ===============================================================================
echo.
echo  [Mẹo] Trình duyệt sẽ tự động mở sau 3 giây.
echo  [Mẹo] Để dừng hệ thống, nhấn tổ hợp phím [Ctrl + C] hoặc đóng cửa sổ này.
echo.

:: Tự động mở trình duyệt sau 3 giây khi server sẵn sàng
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"

:: Khởi chạy Master Pipeline của hệ thống
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" --no-gui

echo.
echo [Hệ thống đã dừng an toàn]
pause
