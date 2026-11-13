$procs = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*voice_typing.py*' }

if (-not $procs) {
    Write-Host "语音输入当前未在运行。"
    exit 0
}

foreach ($p in $procs) {
    try {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        Write-Host "已停止进程 PID $($p.ProcessId)"
    } catch {
        Write-Host "停止失败 PID $($p.ProcessId)：$($_.Exception.Message)"
    }
}
