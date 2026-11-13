$procs = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*voice_typing.py*' }

if ($procs) {
    Write-Host "状态：运行中"
    foreach ($p in $procs) { Write-Host ("  PID {0}  {1}" -f $p.ProcessId, $p.Name) }
} else {
    Write-Host "状态：未运行"
}

$gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
if ($gpu) { Write-Host "显卡：$gpu" }

$log = Join-Path $PSScriptRoot "logs\voice_typing.log"
if (Test-Path $log) {
    Write-Host ""
    Write-Host "最近日志："
    Get-Content $log -Tail 10 | ForEach-Object { Write-Host "  $_" }
}
