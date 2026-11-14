$procs = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*voice_typing.py*' }

if ($procs) {
    # venv 的 pythonw.exe 是个引导器存根, 它会拉起基础解释器(py312)当子进程干活,
    # 所以一个实例对应两个 PID; 只有父进程不在列表里的那个才算独立实例
    $ids = @($procs | ForEach-Object { $_.ProcessId })
    $roots = @($procs | Where-Object { $ids -notcontains $_.ParentProcessId })
    Write-Host ("状态：运行中 ({0} 个实例)" -f $roots.Count)
    foreach ($p in $procs) {
        $tag = if ($ids -contains $p.ParentProcessId) { "  (venv 引导器拉起的主进程)" } else { "  (引导器存根)" }
        Write-Host ("  PID {0}  {1}{2}" -f $p.ProcessId, $p.Name, $tag)
    }
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
