"""Run one serving phase, retaining logs and always cleaning up its processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('/workspace/disagg'))
p.add_argument('--mode',choices=['replicas','pd'],required=True)
p.add_argument('--smoke-only',action='store_true')
p.add_argument('--diagnose-only',action='store_true')
p.add_argument('--variant',choices=['default','prefill16k','batch16k','batch1k','balanced','kv128','kv128replicas','shapes'],default='default')
a=p.parse_args()
assert a.variant not in ('prefill16k','kv128') or a.mode=='pd'
assert a.variant not in ('batch16k','batch1k','balanced','kv128replicas') or a.mode=='replicas'
python='/opt/disagg/venv/bin/python'
raw=a.root/'raw'
if a.variant!='default':raw=raw/a.variant
raw.mkdir(parents=True,exist_ok=True)
procs=[]; logs=[]
label=a.mode+('-smoke' if a.smoke_only else '-diagnostic' if a.diagnose_only else '')

def spawn(cmd,name):
    log=(raw/f'{label}-{name}.log').open('w'); logs.append(log)
    proc=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
    procs.append(proc)
    return proc

def ready(proc,url):
    until=time.monotonic()+900
    while time.monotonic()<until:
        if proc.poll() is not None: raise RuntimeError(f'Process exited {proc.returncode}: {url}')
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                if response.status==200: return
        except Exception: pass
        time.sleep(2)
    raise TimeoutError('Worker startup exceeded 15 minutes')

try:
    roles=['replica','replica'] if a.mode=='replicas' else ['prefill','decode']
    workers=[]
    for gpu,(role,port) in enumerate(zip(roles,(8100,8200))):
        batch_tokens=4096
        if a.variant=='batch16k' or (a.variant=='prefill16k' and role=='prefill'):batch_tokens=16384
        if a.variant=='batch1k':batch_tokens=1024
        workers.append(spawn([python,str(a.root/'worker.py'),'--root',str(a.root),
            '--gpu',str(gpu),'--port',str(port),'--role',role,'--batch-tokens',str(batch_tokens),'--block-size','16' if a.variant=='default' else '128']+(['--ucx-proto-info'] if a.diagnose_only or a.variant!='default' else []),f'worker{gpu}'))
    for worker,port in zip(workers,(8100,8200)): ready(worker,f'http://127.0.0.1:{port}/v1/models')
    router=spawn([python,str(a.root/'router.py'),'--mode',a.mode,'--events',str(raw/f'{label}-router-events.jsonl'),'--balance','prompt_tokens' if a.variant=='balanced' else 'requests'],'router')
    ready(router,'http://127.0.0.1:8000/health')
    print(label+' workers and router ready',flush=True)
    if a.diagnose_only:
        assert a.mode=='pd'
        subprocess.run([python,str(a.root/'diagnose.py'),'--root',str(a.root)],check=True,timeout=600)
    elif a.smoke_only:
        payload={'model':'qwen3','prompt':'Explain why a GPU KV cache is useful.','temperature':0,'max_tokens':32,'ignore_eos':True,'stream':False}
        req=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as r: result=json.load(r)
        assert result['usage']['completion_tokens']==32
        (raw/'smoke-response.json').write_text(json.dumps(result,indent=2))
        time.sleep(3)
        for port in (8100,8200):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics') as r:
                (raw/f'smoke-metrics-{port}.txt').write_bytes(r.read())
        (a.root/'SMOKE_READY').touch()
        print('SMOKE_READY; keeping workers alive briefly for external verification',flush=True)
        until=time.monotonic()+300
        while not (a.root/'EXTERNAL_SMOKE_COMPLETE').exists():
            if time.monotonic()>until: raise TimeoutError('External verification not completed in five minutes')
            time.sleep(2)
    else:
        command=[python,str(a.root/'benchmark.py'),'--root',str(a.root),'--mode',a.mode,'--raw',str(raw)]
        if a.variant=='shapes':command+=['--profiles','1k','8k','32k','--concurrencies','16','--requests-per-run','32','--shape-suite']
        elif a.variant!='default':command+=['--profiles','32k','mixed','--concurrencies','16']
        subprocess.run(command,check=True,timeout=3600)
    if a.diagnose_only or a.variant=='kv128':
        subprocess.run([python,str(a.root/'transfer_probe.py'),'--root',str(a.root),'--output',str(raw/('transfer-probe-default.json' if a.diagnose_only else 'transfer-probe.json'))],check=True,timeout=600)
    (a.root/((label+'-'+a.variant if a.variant!='default' else label).upper()+'_COMPLETE')).touch()
finally:
    for proc in reversed(procs):
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
    for log in logs: log.close()
