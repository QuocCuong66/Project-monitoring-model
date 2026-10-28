@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Fall and Stroke Warning System — Build Release
cls

:: Tham số (tuỳ chọn) để CI chèn bước ký số giữa các giai đoạn:
::   (trống)    chạy đủ 3 bước
::   app        chỉ bước 1 — PyInstaller
::   installer  chỉ bước 2 — Inno Setup (dùng dist\%APP%\ đã có)
set "STAGE=%~1"

set "APP=Fall_and_Stroke_Warning_System"
set "APPDIR=dist\%APP%"

echo ===============================================================================
echo     FALL AND STROKE WARNING SYSTEM — BUILD BẢN PHÁT HÀNH
echo ===============================================================================
echo  Kết quả trong thư mục dist\ :
echo   1. %APP%\                       (ứng dụng độc lập .exe)
echo   2. %APP%_Setup_v*.exe           (bộ cài đặt cho người dùng)
echo   3. %APP%_Source_v*.zip          (mã nguồn)
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy môi trường ảo .venv!
    pause
    exit /b 1
)

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo [*] Đang cài đặt PyInstaller vào .venv...
    .\.venv\Scripts\python.exe -m pip install pyinstaller
)

:: Phiên bản lấy từ pyproject.toml
for /f "usebackq delims=" %%v in (`.\.venv\Scripts\python.exe -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"`) do set "APP_VERSION=%%v"
echo [*] Phiên bản: %APP_VERSION%
echo.
if /i "%STAGE%"=="installer" goto :installer

:: ---------------------------------------------------------------------------
:: 1. PyInstaller
:: ---------------------------------------------------------------------------
echo [1/3] Biên dịch ứng dụng bằng PyInstaller (mất vài phút)...
.\.venv\Scripts\pyinstaller.exe --clean "fall_and_stroke_monitor.spec" -y --log-level WARN
if %errorlevel% neq 0 goto :fail

:: Cấu hình + giao diện đặt cạnh file .exe để người dùng chỉnh sửa được
:: (mô hình AI nằm trong _internal\models, không sao chép trùng lặp)
xcopy /E /I /Y /Q "configs" "%APPDIR%\configs" >nul
xcopy /E /I /Y /Q "frontend" "%APPDIR%\frontend" >nul
if not exist "%APPDIR%\scripts" mkdir "%APPDIR%\scripts"
copy /Y "scripts\cai_dat_startup.vbs" "%APPDIR%\scripts\" >nul
copy /Y "scripts\huy_startup.vbs" "%APPDIR%\scripts\" >nul
copy /Y "app_icon.ico" "%APPDIR%\" >nul
copy /Y "HUONG_DAN_SU_DUNG.txt" "%APPDIR%\" >nul
if not exist "%APPDIR%\data\processed\events" mkdir "%APPDIR%\data\processed\events"
if /i "%STAGE%"=="app" goto :done

:: ---------------------------------------------------------------------------
:: 2. Bộ cài đặt Inno Setup
:: ---------------------------------------------------------------------------
:installer
echo [2/3] Tạo bộ cài đặt Inno Setup...
set "ISCC="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
    echo [LỖI] Chưa cài Inno Setup 6. Cài bằng lệnh:
    echo        winget install --id JRSoftware.InnoSetup -e --scope user
    goto :fail
)
"%ISCC%" /Q /DMyAppVersion=%APP_VERSION% "installer\setup.iss"
if %errorlevel% neq 0 goto :fail
if /i "%STAGE%"=="installer" goto :done

:: ---------------------------------------------------------------------------
:: 3. Gói mã nguồn
:: ---------------------------------------------------------------------------
echo [3/3] Đóng gói mã nguồn...
.\.venv\Scripts\python.exe dong_goi_du_an.py
if %errorlevel% neq 0 goto :fail

:done
echo.
echo ===============================================================================
echo  [THÀNH CÔNG] Gửi cho người dùng file:
echo   👉 dist\%APP%_Setup_v%APP_VERSION%.exe
echo  Mã nguồn: dist\%APP%_Source_v%APP_VERSION%.zip
echo ===============================================================================
echo.
if not defined NO_PAUSE pause
exit /b 0

:fail
echo.
echo ===============================================================================
echo  [THẤT BẠI] Build không thành công — xem lỗi ở trên.
echo ===============================================================================
if not defined NO_PAUSE pause
exit /b 1
