' ==============================================================================
' HỦY KHỞI ĐỘNG CÙNG WINDOWS (REMOVE STARTUP SHORTCUT)
' ==============================================================================
Option Explicit
Dim WshShell, fso, startupFolder, shortcutFile

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

startupFolder = WshShell.SpecialFolders("Startup")
shortcutFile = startupFolder & "\FallAndStrokeWarningSystem.lnk"

If fso.FileExists(shortcutFile) Then
    fso.DeleteFile shortcutFile, True
    WScript.Echo "[THÀNH CÔNG] Đã hủy khởi động cùng Windows!"
Else
    WScript.Echo "[THÔNG BÁO] Hệ thống hiện không được cài đặt khởi động cùng Windows."
End If
