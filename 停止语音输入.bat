@echo off
chcp 65001 >nul
title 停止语音输入
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
echo.
timeout /t 2 >nul
exit
