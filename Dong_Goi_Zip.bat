@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Đóng Gói File ZIP
cls

echo ===============================================================================
echo     FALL AND STROKE WARNING SYSTEM — ĐÓNG GÓI MÃ NGUỒN (dist\*_Source_v*.zip)
echo ===============================================================================
echo.
echo [*] Đang đóng gói mã nguồn vào thư mục dist...
echo.

.\.venv\Scripts\python.exe dong_goi_du_an.py

echo.
pause
