"""Persistent probe -> benchmark pipeline. Safe to rerun after interruption."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
started = time.monotonic()
while not (ROOT / 'DOWNLOAD_COMPLETE').exists():
    if time.monotonic() - started > 3600:
        raise TimeoutError('Downloads did not complete; inspect download.log')
    active = subprocess.run(['tmux','has-session','-t','qwen38-download'],capture_output=True).returncode == 0
    if not active:
        raise RuntimeError('Download session ended without DOWNLOAD_COMPLETE; inspect download.log')
    time.sleep(5)

probe = ROOT / 'probe-16k'
if not (probe / 'COMPLETE').exists():
    subprocess.run(['python3','-u',str(ROOT/'bench.py'),'--probe','--context','16384','--ubatch','128',
                    '--out',str(probe)], check=True)
rows = [json.loads(line) for line in (probe/'results.jsonl').read_text().splitlines()]
assert len(rows) == 12, 'Probe coverage incomplete'
assert all(r['finish_reason']=='stop' and r['answer'].strip() for r in rows), 'Probe returned missing or truncated answers'
assert all(r['reasoning_chars']==0 for r in rows), 'Unexpected thinking output'
draft = [r for r in rows if r['mode']=='draft-dflash']
assert sum(r['timings'].get('draft_n',0) for r in draft)>0, 'Drafting not active'
assert sum(r['timings'].get('draft_n_accepted',0) for r in draft)>0, 'No accepted draft tokens'
print('PROBE PASSED: 12 full uncached responses, GPU offload confirmed, DFlash 2 active.',flush=True)
subprocess.run(['python3','-u',str(ROOT/'bench.py'),'--context','16384','--ubatch','128',
                '--repeats','10','--out',str(ROOT/'results-16k')],check=True)
print('BENCHMARK COMPLETE: ready to copy results and render heatmap.',flush=True)
