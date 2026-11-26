@echo off
cd /d "%~dp0"
title Fall and Stroke Warning System — Disable Startup
cls

echo ===============================================================================
echo   FALL AND STROKE WARNING SYSTEM - HUY TU DONG CHAY NEN CUNG WINDOWS
echo ===============================================================================
echo.
echo [*] Dang go bo khoi dong cung Windows...
echo.

if exist "%~dp0huy_startup.vbs" (
    cscript.exe //nologo "%~dp0huy_startup.vbs"
) else (
    cscript.exe //nologo "%~dp0scripts\huy_startup.vbs"
)

echo.
echo ===============================================================================
echo  He thong se khong con tu dong khoi dong khi ban mo may tinh nua.
echo  Ban co the bat lai bat ky luc nao bang file: Bat_Khoi_Dong_Cung_Windows.bat
echo ===============================================================================
echo.
pause
