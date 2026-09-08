"""Audit completed block samples and compare equal-work GEMM scheduling strategies."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
assert (args.run/'COMPLETE').exists()
m=json.loads((args.run/'manifest.json').read_text());cfg=m['config'];rows=json.loads((args.run/'summary.json').read_text())
raw=[json.loads(s) for s in (args.run/'samples.jsonl').read_text().splitlines()]
assert len(rows)==len(raw)==cfg['repeats']*len(cfg['rows'])*6
seen=set()
for i,(r,s) in enumerate(zip(rows,raw)):
 key=(r['repeat'],r['rows'],r['strategy'],r['mode']);assert key not in seen;seen.add(key)
 a=np.array(s['seconds']);assert s['case']==i and len(a)==r['samples']>=cfg['min_samples']
 assert np.isfinite(a).all() and (a>0).all() and a.sum()>=cfg['seconds']
 assert np.isclose(a.sum(),r['measured_seconds'],rtol=1e-10)
 assert np.isclose(np.median(a)/r['count'],r['median_seconds'],rtol=1e-10)
 assert np.isclose(np.percentile(a,95)/r['count'],r['p95_seconds'],rtol=1e-10)
 assert r['total_operations']==r['count']*len(a) and r['correctness']=='PASS'
args.out.mkdir(parents=True,exist_ok=True)
for name in ['manifest.json','summary.json','COMPLETE']:(args.out/name).write_bytes((args.run/name).read_bytes())
audit=dict(status='PASS',cases=len(rows),samples=sum(r['samples'] for r in rows),request_groups=sum(r['total_operations'] for r in rows),measured_seconds=sum(r['measured_seconds'] for r in rows),raw_sha256=hashlib.sha256((args.run/'samples.jsonl').read_bytes()).hexdigest())
(args.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
with (args.out/'timings.csv').open('w') as f:
 writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True);table=[]
for ax,mode in zip(axs,['ordinary','graph']):
 for strategy,color in zip(['sequential','streams','batched'],['#64748b','#22d3ee','#a78bfa']):
  data=[]
  for n in cfg['rows']:
   vals=[r['median_seconds']*1e6 for r in rows if r['mode']==mode and r['strategy']==strategy and r['rows']==n]
   data.append((n,np.median(vals),min(vals),max(vals)))
   table.append(f'| {mode} | {n} | {strategy} | {np.median(vals):.2f} |')
  a=np.array(data);ax.plot(a[:,0],a[:,1],'o-',label=strategy,color=color);ax.fill_between(a[:,0],a[:,2],a[:,3],alpha=.15,color=color)
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(cfg['rows'],[str(n) for n in cfg['rows']]);ax.set_xlabel('Rows per request · K=N=1024');ax.set_ylabel('Microseconds per group of four requests');ax.set_title('Ordinary calls' if mode=='ordinary' else 'CUDA Graph replay');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Same shared-weight GEMMs · lower is better · shading = repeat range');fig.tight_layout()
plt.rcParams['svg.hashsalt']='compute-streams-v1';fig.savefig(args.out/'latency.png',dpi=170);fig.savefig(args.out/'latency.svg',metadata={'Date':None})
p=args.out/'latency.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
(args.out/'report.md').write_text('''# Sequential, multiple streams, and batching

All numerical checks and independent timing audits passed. Repetitions are
separate measurement passes in one process with shuffled case order. Each group
contains four requests with the same weight matrix. Inputs are already GPU-resident
and laid out contiguously; batch assembly and queueing delay are excluded.
Times include Python submission and block completion, including CUDA Graph replay
times. These are medians of per-group block averages, not request tail latencies.

![Equal-work latency comparison](latency.png)

| Submission | Rows per request | Strategy | Median µs / four-request group |
| --- | ---: | --- | ---: |
'''+ '\n'.join(table)+f'\n\nMeasured {audit["cases"]} cases and {audit["samples"]:,} blocks, covering {audit["request_groups"]:,} four-request groups.\n')
print(json.dumps(audit))
