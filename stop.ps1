$dir = $PSScriptRoot
# 只匹配本目录的常驻实例: 排除 --file/--dry-run/--unmute 等一次性进程, 别打断它们干活
$procs = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*$dir*voice_typing.py*" -and
                   $_.CommandLine -notmatch '--(file|dry-run|unmute)' }

if (-not $procs) {
    Write-Host "语音输入当前未在运行。"
    Remove-Item -ErrorAction SilentlyContinue (Join-Path $dir "logs\stop.flag")
    exit 0
}

# 优雅停止: 写标志让程序自己恢复扬声器再退出。
# 直接强杀会让扬声器永远留在静音态(程序没机会恢复), 之前发生过
$flag = Join-Path $dir "logs\stop.flag"
New-Item -ItemType File -Path $flag -Force | Out-Null
$ids = @($procs | ForEach-Object { $_.ProcessId })
$deadline = (Get-Date).AddSeconds(5)
do {
    Start-Sleep -Milliseconds 200
    $alive = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
} while ($alive.Count -gt 0 -and (Get-Date) -lt $deadline)

foreach ($p in $alive) {
    try {
        Stop-Process -Id $p -Force -ErrorAction Stop
        Write-Host "未响应优雅停止, 强制停止 PID $p"
    } catch {
        Write-Host "停止失败 PID $p：$($_.Exception.Message)"
    }
}

# 强杀兜底后, 若扬声器静音是本程序遗留的, 就地补恢复
if ($alive.Count -gt 0) {
    $py = Join-Path $dir "venv\Scripts\python.exe"
    if (Test-Path $py) {
        & $py (Join-Path $dir "voice_typing.py") --unmute 2>$null
    }
}

Remove-Item -ErrorAction SilentlyContinue $flag
Write-Host "已停止语音输入。"
