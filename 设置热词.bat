@echo off
chcp 65001 >nul
title 设置语音输入热词
cd /d "%~dp0"

echo 即将打开 hotwords.txt，一行一个词，改完保存并关闭记事本即可。
echo.
start /wait notepad "%~dp0hotwords.txt"

echo 正在重启语音输入以加载热词...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
timeout /t 2 >nul
start "" wscript.exe "%~dp0silent_start.vbs"
timeout /t 12 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"
echo.
echo 完成。按任意键关闭。
pause >nul
