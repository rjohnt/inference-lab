"""Low-frequency host telemetry, run uniformly for every benchmark method."""
import json,subprocess,time,threading
from pathlib import Path
FIELDS='timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu,clocks_event_reasons.active'
class Monitor:
    def __init__(self,out):
        self.out=out;self.done=threading.Event();self.phase='startup';self.proc=None
    def start(self):
        self.log=(self.out/'gpu-telemetry.csv').open('w')
        self.proc=subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu='+FIELDS,'--format=csv','--loop-ms=2000'],stdout=self.log,stderr=subprocess.STDOUT)
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
    def sample(self):
        with (self.out/'host-telemetry.jsonl').open('w') as f:
            while not self.done.is_set():
                row={'unix_time':time.time(),'phase':self.phase,'meminfo':Path('/proc/meminfo').read_text(),'loadavg':Path('/proc/loadavg').read_text(),'vmstat':{k:int(v) for k,v in (s.split() for s in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout','pgmajfault')}}
                f.write(json.dumps(row)+'\n');f.flush();self.done.wait(2)
    def snapshot(self,label):
        ps="Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,FreeVirtualMemory,TotalVirtualMemorySize | ConvertTo-Json; Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage | ConvertTo-Json; Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 8 Name,WorkingSet64 | ConvertTo-Json"
        try:
            p=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',ps],capture_output=True,text=True,timeout=30)
            (self.out/(label+'-windows.txt')).write_text(p.stdout+p.stderr)
        except Exception as e:(self.out/(label+'-windows.txt')).write_text(str(e))
    def stop(self):
        self.done.set();self.thread.join(timeout=5)
        if self.proc:self.proc.terminate();self.proc.wait(timeout=5)
        self.log.close()
