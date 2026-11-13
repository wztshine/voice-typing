@echo off
chcp 65001 >nul
title 语音打字 - 一键安装环境
cd /d "%~dp0"

echo ============================================================
echo  语音打字 一键安装
echo  需要: NVIDIA 显卡 + Python 3.10~3.12 (64位)
echo  全程联网, 模型约 5GB, 请耐心等待
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 python, 请先安装 Python 3.10~3.12 并勾选 "Add to PATH"
    pause & exit /b 1
)

if not exist venv (
    echo [1/5] 创建虚拟环境 ...
    python -m venv venv || (echo venv 创建失败 & pause & exit /b 1)
) else (
    echo [1/5] 虚拟环境已存在, 跳过
)

echo [2/5] 安装 PyTorch (CUDA 12.6 版, 约 2.5GB) ...
venv\Scripts\python -m pip install -q -U pip
venv\Scripts\pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (echo PyTorch 安装失败 & pause & exit /b 1)

echo [3/5] 安装其余依赖 ...
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (echo 依赖安装失败 & pause & exit /b 1)

echo [4/5] 下载识别模型 (ModelScope, 约 5GB) ...
venv\Scripts\python dl.py
if errorlevel 1 (echo 模型下载失败, 可重新运行本脚本续传 & pause & exit /b 1)

echo [5/5] 转换 fp16 权重 (省一半显存, 加载更快) ...
venv\Scripts\python make_fp16.py

if not exist hotwords.txt copy hotwords.example.txt hotwords.txt >nul

echo.
echo ============================================================
echo  安装完成! 双击「启动语音输入.bat」即可使用
echo  (首次启动要加载模型, 等 20 秒左右出现"就绪")
echo ============================================================
pause
