"""Compare targeted tuning against the original, equal-hardware controls."""
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
p=argparse.ArgumentParser();p.add_argument('raw',type=Path)
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/tuning');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
variants=[('replicas4k','replicas','', 'Replicas\n4K','#168aad'),
          ('pd4k','pd','', 'P/D\n4K','#a44aab'),
          ('kv128','pd','kv128','P/D\nKV=128','#bb79c1'),
          ('kv128replicas','replicas','kv128replicas','Replicas\nKV=128','#35b4c7'),
          ('prefill16k','pd','prefill16k','P/D\n16K prefill','#6d2f7b'),
          ('batch16k','replicas','batch16k','Replicas\n16K','#87c7d4'),
          ('batch1k','replicas','batch1k','Replicas\n1K','#344e81'),
          ('balanced','replicas','balanced','Replicas\nweighted router','#e59a2a')]
fields=['output_tok_s','mean_ttft_ms','mean_tpot_ms','p95_chunk_gap_ms']
runs=[];summary=[];checks={};validation={}
for name,mode,subdir,label,color in variants:
    folder=a.raw/subdir
    for profile in ('32k','mixed'):
        cell={'variant':name,'mode':mode,'profile':profile,'concurrency':16}
        batch=[]
        for rep in (1,2,3):
            d=json.loads((folder/f'{mode}-{profile}-c16-r{rep}.json').read_text())
            assert len(d['requests'])==16 and all(r['output_tokens']==128 for r in d['requests'])
            d['p95_chunk_gap_ms']=float(np.percentile([v for r in d['requests'] for v in r['chunk_gaps_ms']],95))
            row={'variant':name,'mode':mode,'profile':profile,'concurrency':16,'rep':rep,**{k:d[k] for k in fields}}
            runs.append(row);batch.append(row)
        for key in fields:cell[key]={'median':st.median(r[key] for r in batch),'min':min(r[key] for r in batch),'max':max(r[key] for r in batch)}
        summary.append(cell)
    if subdir:
        c=json.loads((folder/'correctness-comparison.json').read_text());assert len(c)==8;checks[name]=c
        events=[json.loads(x) for x in (folder/f'{mode}-router-events.jsonl').read_text().splitlines()]
        assert len(events)==136,(name,len(events))
        if mode=='pd':assert all(e['transfer_metadata'] for e in events)
        counters={}
        m=json.loads((folder/f'{mode}-final-metrics.json').read_text())
        for gpu,port in enumerate(('8100','8200')):
            values={}
            for line in m[port].splitlines():
                match=re.match(r'([^#{ ]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
                if not match:continue
                key,value=match.groups()
                if key=='vllm:num_preemptions_total' or key.startswith('vllm:nixl_') and not key.endswith(('_created','_bucket')):
                    values[key]=values.get(key,0)+float(value)
            counters[f'gpu{gpu}']=values
        validation[name]={'completed_requests_including_warmup_and_checks':len(events),'requests_with_transfer_metadata':sum(e['transfer_metadata'] for e in events),'metrics':counters}
with (a.output/'runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(runs[0]),lineterminator='\n');w.writeheader();w.writerows(runs)
for filename,value in [('summary',summary),('correctness',checks),('validation',validation)]:
    (a.output/f'{filename}.json').write_text(json.dumps(value,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
for filename,names,subtitle in [
    ('cache-blocks',['replicas4k','pd4k','kv128replicas','kv128'],'KV cache block size: 16 → 128 tokens; batch budget fixed at 4K'),
    ('scheduling',['kv128replicas','kv128','prefill16k','batch16k','batch1k','balanced'],'Scheduling controls; all use 128-token KV blocks')]:
    chosen=[next(v for v in variants if v[0]==name) for name in names]
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    for axrow,profile in zip(axes,('32k','mixed')):
        for ax,key,title,divisor in zip(axrow,fields[:3],['Output tokens/s ↑','Time to first text (s) ↓','Time per output token (ms) ↓'],[1,1000,1]):
            rows=[next(c for c in summary if c['variant']==v[0] and c['profile']==profile) for v in chosen]
            mid=np.array([c[key]['median']/divisor for c in rows]);lo=np.array([c[key]['min']/divisor for c in rows]);hi=np.array([c[key]['max']/divisor for c in rows])
            ax.bar(np.arange(len(chosen)),mid,color=[v[4] for v in chosen],yerr=[mid-lo,hi-mid],capsize=3)
            ax.set_xticks(np.arange(len(chosen)),[v[3] for v in chosen],rotation=25,ha='right',fontsize=9)
            ax.set_title(f'{profile} · {title}',loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Qwen3-8B BF16 · same 2 × A100 80GB · concurrency 16\n'+subtitle+'\n128 output tokens · medians and min–max across 3 runs',fontsize=14)
    for ext in ('png','svg'):
        path=a.output/f'{filename}.{ext}';fig.savefig(path,dpi=160,metadata={'Creator':'inference-lab'} if ext=='svg' else {'Software':'inference-lab'})
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    plt.close(fig)
for row in summary:print(row['variant'],row['profile'],{k:round(row[k]['median'],2) for k in fields})
