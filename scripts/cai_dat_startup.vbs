' ==============================================================================
' TỰ ĐỘNG THIẾT LẬP KHỞI ĐỘNG CÙNG WINDOWS (STARTUP SHORTCUT CREATOR)
' ==============================================================================
Option Explicit
Dim WshShell, fso, startupFolder, currentDir, shortcut, iconPath

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
Dim rootDir, exePath
If fso.FileExists(currentDir & "\Fall_and_Stroke_Warning_System.exe") Then
    rootDir = currentDir
Else
    rootDir = fso.GetParentFolderName(currentDir)
End If

startupFolder = WshShell.SpecialFolders("Startup")

Set shortcut = WshShell.CreateShortcut(startupFolder & "\FallAndStrokeWarningSystem.lnk")
exePath = rootDir & "\Fall_and_Stroke_Warning_System.exe"

If fso.FileExists(exePath) Then
    shortcut.TargetPath = exePath
    shortcut.WorkingDirectory = rootDir
Else
    shortcut.TargetPath = "wscript.exe"
    shortcut.Arguments = Chr(34) & currentDir & "\Chay_Ngam_He_Thong.vbs" & Chr(34)
    shortcut.WorkingDirectory = rootDir
End If

iconPath = rootDir & "\app_icon.ico"
If fso.FileExists(iconPath) Then
    shortcut.IconLocation = iconPath & ",0"
End If

shortcut.Description = "Fall and Stroke Warning System - AI Safety Surveillance"
shortcut.Save

WScript.Echo "[THÀNH CÔNG] Đã đăng ký khởi động ngầm cùng Windows!"
WScript.Echo "Vị trí: " & startupFolder & "\FallAndStrokeWarningSystem.lnk"
