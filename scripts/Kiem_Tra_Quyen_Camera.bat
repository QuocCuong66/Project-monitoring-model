@echo off
chcp 65001 >nul
if exist "%~dp0..\main.py" (
    cd /d "%~dp0.."
) else (
    cd /d "%~dp0"
)
title Fall and Stroke Warning System — Kiểm Tra Quyền & Khả Dụng Camera
cls

echo ===============================================================================
echo   FALL AND STROKE WARNING SYSTEM — KIỂM TRA QUYỀN TRUY CẬP CAMERA
echo ===============================================================================
echo.
echo [*] Đang kiểm tra các thiết bị camera trên máy tính...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "src\camera\camera_permission.py"
) else if exist "runtime\python.exe" (
    "runtime\python.exe" "src\camera\camera_permission.py"
) else (
    python "src\camera\camera_permission.py"
)

echo.
echo ===============================================================================
echo  Nhấn phím bất kỳ để kết thúc kiểm tra.
echo ===============================================================================
pause
