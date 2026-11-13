@echo off
chcp 65001 >nul
title 设置语音输入热键
cd /d "%~dp0"

"%~dp0venv\Scripts\python.exe" pick_key.py %*
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

echo.
echo 正在重启语音输入以应用新热键...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
timeout /t 2 >nul
start "" wscript.exe "%~dp0silent_start.vbs"
timeout /t 10 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"
echo.
echo 完成。按任意键关闭。
pause >nul
