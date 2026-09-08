"""Run the pinned vLLM baseline, first decode-only, then real serving."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--phase',choices=['decode','api','both'],default='both')
a=p.parse_args()
root=a.root
raw=root/'raw'
raw.mkdir(exist_ok=True)
venv=(root/'vllm-venv').resolve()
python=str(venv/'bin/python')
vllm=[python,'-m','vllm.entrypoints.cli.main']
base=vllm+['serve',str(root/'checkpoint'),'--served-model-name','north-mini',
    '--host','127.0.0.1','--port','8000','--dtype','bfloat16',
    '--max-model-len','33792','--max-num-seqs','8','--max-num-batched-tokens','4096',
    '--gpu-memory-utilization','0.90','--seed','42',
    '--attention-config',json.dumps({'backend':'FLASH_ATTN','flash_attn_version':3}),
    '--kernel-config',json.dumps({'moe_backend':'triton'})]

def ready(proc):
    until=time.monotonic()+1200
    while time.monotonic()<until:
        if proc.poll() is not None:
            raise RuntimeError(f'Server exited: {proc.returncode}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2) as r:
                if r.status==200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError('Server startup exceeded 20 minutes')

for phase in (['decode','api'] if a.phase=='both' else [a.phase]):
    cmd=list(base)
    if phase=='decode':
        cmd+=['--no-enable-prefix-caching']
        cmd+=['--kv-transfer-config',json.dumps({'kv_connector':'DecodeBenchConnector',
            'kv_role':'kv_both','kv_connector_extra_config':{'fill_mean':0.0,'fill_std':0.01}})]
    else:
        cmd+=['--enable-prefix-caching']
    with (raw/f'vllm-{phase}-server.log').open('w') as log:
        telemetry=(raw/f'vllm-{phase}-gpu.csv').open('w')
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,nounits','--loop=1'],stdout=telemetry)
        proc=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
        try:
            ready(proc)
            if phase=='api':
                subprocess.run([python,str(root/'bench_api.py'),'--checkpoint',str(root/'checkpoint'),
                    '--engine','vllm','--output',str(raw/'api-vllm.json')],check=True)
            else:
                subprocess.run([python,str(root/'bench_vllm.py'),'--root',str(root)],check=True,timeout=3600)
        finally:
            monitor.terminate()
            monitor.wait()
            telemetry.close()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
    (root/f'VLLM_{phase.upper()}_COMPLETE').touch()
