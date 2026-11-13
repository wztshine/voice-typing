' 双击即可：后台静默启动语音输入（无任何窗口）
Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = base

py = base & "\venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then
    MsgBox "找不到 " & py & vbCrLf & "请检查 G:\AZ\stt 是否完整。", 16, "语音输入"
    WScript.Quit 1
End If

sh.Run """" & py & """ """ & base & "\voice_typing.py""", 0, False
