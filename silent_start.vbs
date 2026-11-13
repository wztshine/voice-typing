' 后台静默启动语音输入 (无控制台窗口)
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "G:\AZ\stt"
sh.Run """G:\AZ\stt\venv\Scripts\pythonw.exe"" ""G:\AZ\stt\voice_typing.py""", 0, False
