"""Export an explicitly provisional snapshot of completed serving measurements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser();p.add_argument('raw',type=Path);a=p.parse_args()
out=Path(__file__).resolve().parents[1]/'results';out.mkdir(exist_ok=True)
runs=[]
for mode in ('tp','pp','ep'):
    for path in sorted((a.raw/mode).glob('*-r[123]-run.json')):
        meta=json.loads(path.read_text())
        data=json.loads(path.with_name(path.name.replace('-run.json','.json')).read_text())
        expected={'decode':1024,'intermediate':512,'prefill':128}[meta['profile']]
        assert data['completed']==meta['num_prompts'] and data['failed']==0
        assert data['total_output_tokens']==expected*data['completed']
        def preemptions(text):
            return sum(float(x) for x in re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.e+]+)$',text,re.M))
        runs.append({**{k:meta[k] for k in ('mode','profile','concurrency','rep')},
            **{k:data[k] for k in ('completed','failed','total_input_tokens','total_output_tokens','duration','output_throughput','mean_ttft_ms','mean_tpot_ms')},
            'preemptions':preemptions(meta['metrics_after'])-preemptions(meta['metrics_before'])})
cells=[]
for mode,profile,c in sorted(set((r['mode'],r['profile'],r['concurrency']) for r in runs)):
    group=[r for r in runs if (r['mode'],r['profile'],r['concurrency'])==(mode,profile,c)]
    if len(group)!=3:continue
    cells.append({'mode':mode,'profile':profile,'concurrency':c,
        **{k:{'median':statistics.median(r[k] for r in group),'min':min(r[k] for r in group),'max':max(r[k] for r in group)} for k in ('output_throughput','mean_ttft_ms','mean_tpot_ms','preemptions')}})
snapshot={'status':'in progress; provisional snapshot, final archive and cross-mode analysis pending',
    'captured_utc':datetime.now(timezone.utc).isoformat(),'measured_runs':len(runs),
    'timed_requests':sum(r['completed'] for r in runs),'planned_measured_runs':81,
    'planned_timed_requests':2592,'completed_cells':cells,'runs':runs}
(out/'progress.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print(snapshot['measured_runs'],'runs;',snapshot['timed_requests'],'timed requests')
for r in cells:
    print(f"| {r['mode'].upper()} | {r['profile']} | {r['concurrency']} | {r['output_throughput']['median']:.2f} | {r['mean_ttft_ms']['median']/1000:.3f} | {r['mean_tpot_ms']['median']:.2f} | {r['preemptions']['median']:.0f} |")
