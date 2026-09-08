"""Serve the upstream megakernel and run the shared real-prompt client."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
root=a.root
python=str((root/'mk-venv').resolve()/'bin/python')
with (root/'raw/mk-api-server.log').open('w') as log:
    telemetry=(root/'raw/mk-api-gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
        '--format=csv,nounits','--loop=1'],stdout=telemetry)
    cmd=[python,str(root/'megakernel/src/serving/server.py'),'--host','127.0.0.1',
        '--port','8000','--lib',str(root/'megakernel/build/libmk_release.so'),
        '--ckpt',str(root/'checkpoint'),'--device','cuda:0','--bs','8',
        '--frac-vram-utilization','0.90','--max-context','33792',
        '--prefill-chunk-size','4096','--num-sms','132']
    proc=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
    try:
        deadline=time.monotonic()+1200
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'Megakernel server exited: {proc.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=2) as response:
                    if response.status==200:
                        import json
                        model=json.load(response)['data'][0]['id']
                        break
            except Exception:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('Megakernel server startup exceeded 20 minutes')
            time.sleep(3)
        subprocess.run([python,str(root/'bench_api.py'),'--checkpoint',str(root/'checkpoint'),
            '--engine','megakernel','--model',model,'--output',str(root/'raw/api-megakernel.json')],check=True)
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
(root/'MK_API_COMPLETE').touch()
