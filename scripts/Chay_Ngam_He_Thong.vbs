' ==============================================================================
' FALL AND STROKE WARNING SYSTEM — CHẠY ẨN DƯỚI NỀN (BACKGROUND SILENT RUNNER)
' Tự động khởi chạy hệ thống AI mà không làm hiện cửa sổ Command Prompt đen.
' ==============================================================================
Option Explicit
Dim WshShell, fso, currentDir, pythonExe, mainPy, appIcon

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
Dim rootDir
If fso.FileExists(currentDir & "\main.py") Then
    rootDir = currentDir
Else
    rootDir = fso.GetParentFolderName(currentDir)
End If

' 1. Tìm kiếm Python Runner (ưu tiên pythonw.exe để không hiện console)
pythonExe = rootDir & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonExe) Then
    pythonExe = rootDir & "\runtime\pythonw.exe"
End If
If Not fso.FileExists(pythonExe) Then
    pythonExe = rootDir & "\.venv\Scripts\python.exe"
End If
If Not fso.FileExists(pythonExe) Then
    pythonExe = "pythonw.exe"
End If

mainPy = rootDir & "\main.py"

' 2. Khởi chạy hệ thống ngầm (0 = Ẩn hoàn toàn cửa sổ, False = Không chặn luồng)
WshShell.CurrentDirectory = rootDir
WshShell.Run Chr(34) & pythonExe & Chr(34) & " " & Chr(34) & mainPy & Chr(34) & " --no-gui", 0, False

' 3. Đợi 3 giây cho server sẵn sàng và mở Web Dashboard
WScript.Sleep 3000
WshShell.Run "http://localhost:8000", 1, False
