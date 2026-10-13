; =============================================================================
;  Fall and Stroke Warning System — Windows Installer (Inno Setup 6)
;  Build: chạy Build_App_PyInstaller.bat (tự gọi ISCC sau khi PyInstaller xong)
;  Thủ công: ISCC.exe installer\setup.iss
; =============================================================================

#define MyAppName "Fall and Stroke Warning System"
#define MyAppShortName "Fall_and_Stroke_Warning_System"
#ifndef MyAppVersion
  #define MyAppVersion "0.3.3"
#endif
#define MyAppPublisher "Quốc Cường"
#define MyAppExeName "Fall_and_Stroke_Warning_System.exe"
#define SourceDir "..\dist\Fall_and_Stroke_Warning_System"

[Setup]
AppId={{6E2B8F3A-4C1D-4F7B-9A52-3D8E1F0C7B44}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Cài vào thư mục người dùng (không cần quyền Admin). Ứng dụng ghi log, ảnh cảnh báo
; và cấu hình camera ngay trong thư mục cài đặt nên không thể đặt trong Program Files.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename={#MyAppShortName}_Setup_v{#MyAppVersion}
SetupIconFile=..\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "vietnamese"; MessagesFile: "Vietnamese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
vietnamese.AutoStart=Tự động chạy bảo vệ khi khởi động Windows
english.AutoStart=Start automatically when Windows starts
vietnamese.LaunchApp=Mở {#MyAppName} ngay
english.LaunchApp=Launch {#MyAppName} now
vietnamese.ReadGuide=Xem hướng dẫn sử dụng
english.ReadGuide=View the user guide

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "{cm:AutoStart}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Cấu hình camera do người dùng tự chỉnh — giữ nguyên khi cập nhật phiên bản mới
Source: "{#SourceDir}\configs\camera.yaml"; DestDir: "{app}\configs"; Flags: onlyifdoesntexist
Source: "{#SourceDir}\*"; DestDir: "{app}"; Excludes: "app_log.txt,configs\camera.yaml,data\processed\events\*"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\data\processed\events"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\{cm:ReadGuide}"; Filename: "{app}\HUONG_DAN_SU_DUNG.txt"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon
; Cùng tên shortcut mà nút "Tự động chạy" trong Cài đặt của ứng dụng sử dụng
Name: "{userstartup}\FallAndStrokeWarningSystem"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchApp}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM {#MyAppExeName}"; Flags: runhidden; RunOnceId: "KillApp"

[UninstallDelete]
Type: files; Name: "{app}\app_log.txt"
Type: files; Name: "{userstartup}\FallAndStrokeWarningSystem.lnk"
