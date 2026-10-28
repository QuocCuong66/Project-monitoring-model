@echo off
cd /d "%~dp0"
title Fall and Stroke Warning System — Dung He Thong
cls

echo ===============================================================================
echo   FALL AND STROKE WARNING SYSTEM - DUNG HE THONG
echo ===============================================================================
echo.
echo [*] Dang gui tin hieu dung he thong an toan...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $res = Invoke-RestMethod -Uri 'http://localhost:8000/api/system/shutdown' -Method Post -TimeoutSec 3; Write-Host '[THÀNH CÔNG] Đã gửi lệnh dừng hệ thống!' -ForegroundColor Green } catch { " ^
  "$portProc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; " ^
  "if ($portProc) { Stop-Process -Id $portProc -Force; Write-Host '[THÀNH CÔNG] Đã kết thúc tiến trình AI thành công!' -ForegroundColor Green } " ^
  "else { Write-Host '[THÔNG BÁO] Hệ thống hiện không hoạt động.' -ForegroundColor Yellow } }"

echo.
echo ===============================================================================
pause
