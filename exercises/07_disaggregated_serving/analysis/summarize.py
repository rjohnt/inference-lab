"""Import reviewed numeric benchmark results; raw logs and text stay private."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('raw',type=Path)
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
keys=['mode','profile','concurrency','rep','wall_s','output_tok_s','mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_chunk_gap_ms']
runs=[]; requests=[]
for mode in ('replicas','pd'):
    for profile in ('1k','8k','32k','mixed'):
        for c in (4,16):
            for rep in (1,2,3):
                d=json.loads((a.raw/f'{mode}-{profile}-c{c}-r{rep}.json').read_text())
                assert len(d['requests'])==16
                assert all(r['output_tokens']==128 for r in d['requests'])
                # Recompute percentiles with linear interpolation from raw timings.
                d['p95_ttft_ms']=float(np.percentile([r['ttft_ms'] for r in d['requests']],95))
                d['p95_chunk_gap_ms']=float(np.percentile([v for r in d['requests'] for v in r['chunk_gaps_ms']],95))
                runs.append({k:d[k] for k in keys})
                for r in d['requests']:
                    requests.append({**{k:d[k] for k in ('mode','profile','concurrency','rep')},
                        **{k:r[k] for k in ('index','prompt_tokens','output_tokens','ttft_ms','latency_ms','tpot_ms','text_chunks')}})
with (a.output/'runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(runs)
with (a.output/'requests.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(requests[0]),lineterminator='\n');w.writeheader();w.writerows(requests)
summary=[]
for profile in ('1k','8k','32k','mixed'):
    for c in (4,16):
        cell={'profile':profile,'concurrency':c}
        for mode in ('replicas','pd'):
            rows=[r for r in runs if r['mode']==mode and r['profile']==profile and r['concurrency']==c]
            cell[mode]={k:{'median':st.median(r[k] for r in rows),'min':min(r[k] for r in rows),'max':max(r[k] for r in rows)} for k in keys[4:]}
        cell['pd_throughput_ratio']=cell['pd']['output_tok_s']['median']/cell['replicas']['output_tok_s']['median']
        summary.append(cell)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
checks=json.loads((a.raw/'correctness-comparison.json').read_text())
assert len(checks)==8
(a.output/'correctness.json').write_text(json.dumps(checks,indent=2)+'\n')
# Only export selected metric names and numeric aggregates, dropping all labels.
def metrics(text):
    result={}
    for line in text.splitlines():
        match=re.match(r'([^#{ ]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
        if match:
            name,value=match.groups()
            if (name.startswith('vllm:nixl_') and not name.endswith(('_bucket','_created'))) or name in ('vllm:num_preemptions_total','vllm:request_success_total'):
                result[name]=result.get(name,0)+float(value)
    return result
connector={}
for mode in ('replicas','pd'):
    raw_metrics=json.loads((a.raw/f'{mode}-final-metrics.json').read_text())
    connector[mode]={f'gpu{i}':metrics(raw_metrics[str(port)]) for i,port in enumerate((8100,8200))}
(a.output/'connector-metrics.json').write_text(json.dumps(connector,indent=2)+'\n')
events={}; router_runs=[]
for mode in ('replicas','pd'):
    rows=[json.loads(x) for x in (a.raw/f'{mode}-router-events.jsonl').read_text().splitlines()]
    assert len(rows)==520,(mode,len(rows))
    if mode=='pd':assert all(x['transfer_metadata'] for x in rows)
    for profile_index,profile in enumerate(('1k','8k','32k','mixed')):
        for ci,c in enumerate((4,16)):
            for rep in (1,2,3):
                offset=8+16*(profile_index*8+ci*4+rep)
                batch=rows[offset:offset+16]
                router_runs.append({'mode':mode,'profile':profile,'concurrency':c,'rep':rep,
                    'mean_prefill_ms':st.mean(r['prefill_ms'] for r in batch) if mode=='pd' else None,
                    'mean_remote_block_count':st.mean(r['remote_block_count'] for r in batch) if mode=='pd' else None,
                    'gpu0_requests':sum(r['worker']==0 for r in batch),
                    'gpu1_requests':sum(r['worker']==1 for r in batch)})
    events[mode]={'completed_requests':len(rows),'requests_with_transfer_metadata':sum(x['transfer_metadata'] for x in rows),'worker_counts':{str(i):sum(x['worker']==i for x in rows) for i in (0,1)}}
(a.output/'router-summary.json').write_text(json.dumps(events,indent=2)+'\n')
with (a.output/'router-runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(router_runs[0]),lineterminator='\n');w.writeheader();w.writerows(router_runs)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'replicas':'#168aad','pd':'#a44aab'}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
labels=[f"{r['profile']}\nc={r['concurrency']}" for r in summary]; x=np.arange(len(summary))
for ax,metric,title in zip(axes.flat,['output_tok_s','mean_ttft_ms','mean_tpot_ms','p95_chunk_gap_ms'],['Output throughput (tokens/s) ↑','Mean time to first text (ms) ↓','Mean time per output token (ms) ↓','P95 client text-chunk gap (ms) ↓']):
    for j,mode in enumerate(('replicas','pd')):
        mid=np.array([r[mode][metric]['median'] for r in summary])
        lo=np.array([r[mode][metric]['min'] for r in summary]);hi=np.array([r[mode][metric]['max'] for r in summary])
        ax.bar(x+(j-.5)*.36,mid,.36,label='Two replicas' if mode=='replicas' else 'Prefill / decode',color=colors[mode],yerr=[mid-lo,hi-mid],capsize=3)
    ax.set_xticks(x,labels);ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(frameon=False)
fig.suptitle('Qwen3-8B BF16 · same 2 × A100-SXM4-80GB\nMedians and min–max across 3 measured runs · 16 requests/run · 128 output tokens',fontsize=14)
for ext in ('png','svg'):
    path=a.output/f'comparison.{ext}';fig.savefig(path,dpi=180,metadata={'Creator':'inference-lab'} if ext=='svg' else {'Software':'inference-lab'})
    if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
print(json.dumps(summary,indent=2))
