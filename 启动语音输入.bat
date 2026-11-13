@echo off
chcp 65001 >nul
title 语音输入 - FireRedASR2
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" voice_typing.py %*
echo.
echo 程序已退出, 按任意键关闭...
pause >nul
