@echo off
cd /d "%~dp0"
title Fall and Stroke Warning System — Auto Startup
cls

echo ===============================================================================
echo   FALL AND STROKE WARNING SYSTEM - BAT TU DONG CHAY NEN CUNG WINDOWS
echo ===============================================================================
echo.
echo [*] Dang thiet lap he thong khoi dong chay ngam cung Windows...
echo.

if exist "%~dp0cai_dat_startup.vbs" (
    cscript.exe //nologo "%~dp0cai_dat_startup.vbs"
) else (
    cscript.exe //nologo "%~dp0scripts\cai_dat_startup.vbs"
)

echo.
echo ===============================================================================
echo  [HOAN TAT] He thong da duoc thiet lap khoi dong ngam cung Windows.
echo  Moi khi bat may tinh:
echo  1. He thong AI se tu dong chay ngam (khong hien cua so den).
echo  2. Camera se tu dong giam sat canh bao te nga va dot quy.
echo  3. Trinh duyet Web Dashboard se tu dong mo tai http://localhost:8000.
echo.
echo  (De huy, ban chi can chay file: Tat_Khoi_Dong_Cung_Windows.bat)
echo ===============================================================================
echo.
pause
