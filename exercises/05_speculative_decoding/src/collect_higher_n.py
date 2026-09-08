"""Collect a specific completed run and render all local artifacts without another manual step."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN='results-20260907T225751Z'
REMOTE='/path/to/data/gpustation-benchmarks/'+RUN
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpustation']
status=ROOT/'higher-n-status.json'
def state(**kw):status.write_text(json.dumps({'run':RUN,'updated_unix':time.time(),**kw},indent=2))
state(status='running')
for attempt in range(240):
    p=subprocess.run(SSH+[f'test -f {REMOTE}/COMPLETE && echo COMPLETE; wc -l {REMOTE}/results.jsonl; pgrep -f "^python3 -u bench.py --repeats 10"'],capture_output=True,text=True,timeout=30)
    if 'COMPLETE' in p.stdout:break
    state(status='running' if p.returncode==0 else 'check_required',remote_output=p.stdout,remote_error=p.stderr)
    if p.returncode==1:
        raise RuntimeError('Benchmark process stopped before COMPLETE; inspect remote log')
    time.sleep(30)
else:raise TimeoutError('Collector timed out after two hours')
subprocess.run(['scp','-r','gpustation:'+REMOTE,str(ROOT)+'/'],check=True)
for script in ['report.py','check_results.py','uncertainty.py','plot_comparison.py','comparison_findings.py']:
    subprocess.run(['python3',str(ROOT/script),str(ROOT/RUN)],check=True)
state(status='complete',image=str(ROOT/RUN/'throughput-comparison.png'),report=str(ROOT/RUN/'report.md'),uncertainty=str(ROOT/RUN/'uncertainty.md'))
