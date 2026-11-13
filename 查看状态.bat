@echo off
chcp 65001 >nul
title 语音输入状态
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status.ps1"
echo.
pause
