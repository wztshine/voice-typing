@echo off
chcp 65001 >nul
title 语音打字 - 一键安装环境
cd /d "%~dp0"

echo ============================================================
echo  语音打字 一键安装
echo  需要: NVIDIA 显卡 + Python 3.10~3.12 (64位)
echo  全程联网, 模型约 6GB, 请耐心等待
echo ============================================================
echo.

rem ---- 找 Python: 先 python, 再 py -3 ----
set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [错误] 找不到 Python, 请先安装 Python 3.10~3.12 并勾选 "Add to PATH"
        pause & exit /b 1
    )
    set "PY=py -3"
)

rem ---- 检查 Python 版本 (3.10~3.12 才有对应的 torch 轮子) ----
%PY% -c "import sys; v=sys.version_info; assert (3,10)<=v[:2]<=(3,12), '需要 Python 3.10~3.12, 当前 %d.%d' % v[:2]"
if errorlevel 1 (
    echo [错误] Python 版本不对, 需要 3.10~3.12 (64位)
    %PY% --version
    pause & exit /b 1
)

rem ---- 检查 NVIDIA 显卡(只警告不拦, 方便以后支持 CPU) ----
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [警告] 没检测到 NVIDIA 显卡/驱动, 装完后识别可能跑不起来
)

if not exist venv (
    echo [1/5] 创建虚拟环境 ...
    %PY% -m venv venv || (echo venv 创建失败 & pause & exit /b 1)
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

echo [4/5] 下载识别模型 (ModelScope, 约 6GB) ...
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
