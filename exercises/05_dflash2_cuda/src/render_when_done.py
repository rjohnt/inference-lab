"""Generate the artifacts on GPUStation even if the client goes away."""
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
out = ROOT / 'results-16k'
started = time.monotonic()
while not (out / 'COMPLETE').exists():
    if time.monotonic() - started > 14400:
        raise TimeoutError('Benchmark did not finish within four hours')
    if subprocess.run(['tmux','has-session','-t','qwen38-benchmark'],capture_output=True).returncode:
        raise RuntimeError('Benchmark session ended without COMPLETE; inspect pipeline.log')
    time.sleep(10)

venv = ROOT / 'plot-env'
if not (venv / 'bin/python').exists():
    subprocess.run(['python3','-m','venv','--copies',str(venv)],check=True)
env = dict(os.environ, PIP_CACHE_DIR=str(ROOT/'pip-cache'), MPLCONFIGDIR=str(ROOT/'matplotlib-cache'))
python = str(venv / 'bin/python')
subprocess.run([python,'-m','pip','install','matplotlib==3.9.4','numpy==2.2.6'],env=env,check=True)
subprocess.run([python,str(ROOT/'analyze.py'),str(out)],env=env,check=True)
(out / 'ARTIFACTS_COMPLETE').write_text('heatmap and report generated\n')
print('ARTIFACTS COMPLETE',out,flush=True)
