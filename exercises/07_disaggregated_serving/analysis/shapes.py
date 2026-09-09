"""Summarize workload-shape controls and actual HTTP-stage overlap."""
import argparse
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics as st
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('raw',type=Path)
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/shapes')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
folder=a.raw/'shapes';summary=[];timelines={};validation={};activity=[];samples=[]
for row in csv.reader((a.raw/'gpu-monitor.csv').open()):
    if len(row)!=6:continue
    stamp=datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
    samples.append({'time':stamp,'gpu':int(row[1]),'util':float(row[2]),'power_w':float(row[5])})
profiles=[('1k',1024,'Decode-heavy\n1K input / 1K output'),('8k',512,'Intermediate\n8K input / 512 output'),('32k',128,'Prefill-heavy\n32K input / 128 output')]
fields=['output_tok_s','mean_ttft_ms','mean_tpot_ms','p95_chunk_gap_ms']
for mode in ('pd','replicas'):
    events=[json.loads(x) for x in (folder/f'{mode}-router-events.jsonl').read_text().splitlines()]
    assert len(events)==392,(mode,len(events))
    if mode=='pd':assert all(e['transfer_metadata'] for e in events)
    counters={}
    for gpu,port in enumerate(('8100','8200')):
        values={}
        for line in json.loads((folder/f'{mode}-final-metrics.json').read_text())[port].splitlines():
            match=re.match(r'([^#{ ]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
            if not match:continue
            key,value=match.groups()
            if key=='vllm:num_preemptions_total' or key.startswith('vllm:nixl_') and not key.endswith(('_created','_bucket')):
                values[key]=values.get(key,0)+float(value)
        counters[f'gpu{gpu}']=values
    validation[mode]={'counters':counters,'completed_requests_including_warmup_and_checks':len(events),'requests_with_transfer_metadata':sum(e['transfer_metadata'] for e in events)}
    for profile,tokens,label in profiles:
        batch=[]
        for rep in (1,2,3):
            d=json.loads((folder/f'{mode}-{profile}-c16-r{rep}.json').read_text())
            assert len(d['requests'])==32 and all(r['output_tokens']==tokens for r in d['requests'])
            d['p95_chunk_gap_ms']=float(np.percentile([v for r in d['requests'] for v in r['chunk_gaps_ms']],95))
            batch.append(d)
            for gpu in (0,1):
                selected=[s for s in samples if s['gpu']==gpu and d['started_unix_s']<=s['time']<=d['started_unix_s']+d['wall_s']]
                assert selected
                activity.append({'mode':mode,'profile':profile,'rep':rep,'gpu':gpu,'samples':len(selected),'mean_gpu_util_pct':st.mean(s['util'] for s in selected),'mean_power_w':st.mean(s['power_w'] for s in selected)})
            if mode=='pd' and rep==2:
                start=d['started_unix_s'];end=start+d['wall_s']
                selected=sorted([e for e in events if start<=e['started_unix_s']<end],key=lambda e:e['started_unix_s'])
                assert len(selected)==32
                rows=[{'request':i,'start_s':e['started_unix_s']-start,'prefill_end_s':e['started_unix_s']-start+e['prefill_ms']/1000,'end_s':e['started_unix_s']-start+e['wall_ms']/1000} for i,e in enumerate(selected)]
                # Exact interval sweep: time when at least one P call and one D call coexist.
                changes=[]
                for r in rows:changes.extend([(r['start_s'],1,0),(r['prefill_end_s'],-1,1),(r['end_s'],0,-1)])
                pc=dc=0;last=0;overlap=0
                for t,dp,dd in sorted(changes):
                    if pc>0 and dc>0:overlap+=t-last
                    pc+=dp;dc+=dd;last=t
                timelines[profile]={'rep':2,'wall_s':d['wall_s'],'both_http_stages_in_flight_s':overlap,'rows':rows}
        summary.append({'mode':mode,'profile':profile,'output_tokens':tokens,'requests_per_run':32,'concurrency':16,**{k:{'median':st.median(d[k] for d in batch),'min':min(d[k] for d in batch),'max':max(d[k] for d in batch)} for k in fields}})
checks={mode:json.loads((folder/f'correctness-{mode}.json').read_text()) for mode in ('pd','replicas')}
agreement=[{'context':x['context'],'index':x['index'],'exact_text_match':x['text']==y['text']} for x,y in zip(checks['pd'],checks['replicas'])]
assert len(agreement)==8
for name,data in [('gpu-activity',activity),('correctness',agreement),('summary',summary),('timelines',timelines),('validation',validation)]:
    (a.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    for ext in ('png','svg'):
        path=a.output/f'{name}.{ext}';fig.savefig(path,dpi=170)
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
for ax,key,title,scale in zip(axes,fields[:3],['Output tokens/s ↑','Time to first text (s) ↓','Time per output token (ms) ↓'],[1,1000,1]):
    for offset,mode,color in [(-.19,'replicas','#168aad'),(.19,'pd','#a44aab')]:
        rows=[next(r for r in summary if r['mode']==mode and r['profile']==profile) for profile,_,_ in profiles]
        mid=np.array([r[key]['median']/scale for r in rows]);lo=np.array([r[key]['min']/scale for r in rows]);hi=np.array([r[key]['max']/scale for r in rows])
        ax.bar(np.arange(3)+offset,mid,.36,color=color,label=mode,yerr=[mid-lo,hi-mid],capsize=3)
    ax.set_xticks(range(3),[x[2] for x in profiles],fontsize=8);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.suptitle('Workload shapes · Qwen3-8B BF16 · same 2 × A100 80GB\n32 requests, concurrency 16 · medians and min–max of three measured runs')
save(fig,'comparison')
fig,axes=plt.subplots(3,1,figsize=(12,12),layout='constrained')
for ax,(profile,_,label) in zip(axes,profiles):
    for r in timelines[profile]['rows']:
        ax.broken_barh([(r['start_s'],r['prefill_end_s']-r['start_s'])],(r['request']-.38,.76),facecolors='#e59a2a',label='Prefill call' if r['request']==0 else None)
        ax.broken_barh([(r['prefill_end_s'],r['end_s']-r['prefill_end_s'])],(r['request']-.38,.76),facecolors='#a44aab',label='Decode call' if r['request']==0 else None)
    ax.invert_yaxis();ax.set_ylabel('Request (arrival order)');ax.set_xlabel('Seconds from measured-run start');ax.set_title(label.replace('\n',' · '),loc='left');ax.grid(axis='x',alpha=.2)
axes[0].legend(loc='upper right',frameon=False)
fig.suptitle('Observed P/D request pipeline · measured repeat 2\nHTTP call intervals include queues and KV transfer; these are not GPU kernel intervals',fontsize=13)
save(fig,'stage-overlap')
print(json.dumps(summary,indent=2))

fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
for ax,mode in zip(axes,('replicas','pd')):
    for gpu,color in [(0,'#e59a2a'),(1,'#a44aab')]:
        vals=[st.mean(r['mean_gpu_util_pct'] for r in activity if r['mode']==mode and r['gpu']==gpu and r['profile']==profile) for profile,_,_ in profiles]
        ax.bar(np.arange(3)+(gpu-.5)*.36,vals,.36,color=color,label=f'GPU {gpu}'+(' (prefill)' if mode=='pd' and gpu==0 else ' (decode)' if mode=='pd' else ''))
    ax.set_xticks(range(3),[x[2] for x in profiles],fontsize=8);ax.set_ylim(0,105);ax.set_ylabel('Mean sampled GPU utilization (%)');ax.set_title(mode);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('GPU activity across workload shapes · concurrency 16\nOne-second monitor samples averaged across three measured runs')
save(fig,'gpu-activity')
