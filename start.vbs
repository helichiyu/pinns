Option Explicit

Dim ws, fso, killCmd
Set ws = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
ws.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

' Kill any previous server instance before starting a new one.
killCmd = "powershell -NoProfile -WindowStyle Hidden -Command " & _
          """Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*server.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"""
ws.Run killCmd, 0, True
WScript.Sleep 300

' Launch the backend server silently (0 = hidden window, False = async).
' The server itself opens the browser once ready and exits when the browser is closed.
ws.Run "D:\anaconda3\envs\use\python.exe backend\server.py", 0, False
