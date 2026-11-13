@echo off
chcp 65001 >nul
title 重启语音输入
echo 正在停止...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
timeout /t 2 >nul
echo 正在后台启动...
start "" wscript.exe "%~dp0silent_start.vbs"
timeout /t 3 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"
timeout /t 8 >nul
exit
