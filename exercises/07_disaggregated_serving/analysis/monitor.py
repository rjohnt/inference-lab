"""Aggregate one-second GPU samples within measured request windows."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results');a=p.parse_args()
samples=[]
for row in csv.reader((a.raw/'gpu-monitor.csv').open()):
    if len(row)!=6:continue
    stamp=datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
    samples.append({'time':stamp,'gpu':int(row[1]),'util':float(row[2]),'mem_util':float(row[3]),'memory_mib':float(row[4]),'power_w':float(row[5])})
rows=[]
for mode in ('replicas','pd'):
    for profile in ('1k','8k','32k','mixed'):
        for c in (4,16):
            for rep in (1,2,3):
                d=json.loads((a.raw/f'{mode}-{profile}-c{c}-r{rep}.json').read_text())
                for gpu in (0,1):
                    selected=[s for s in samples if s['gpu']==gpu and d['started_unix_s']<=s['time']<=d['started_unix_s']+d['wall_s']]
                    assert selected,(mode,profile,c,rep,gpu)
                    rows.append({'mode':mode,'profile':profile,'concurrency':c,'rep':rep,'gpu':gpu,'samples':len(selected),'mean_gpu_util_pct':st.mean(s['util'] for s in selected),'mean_power_w':st.mean(s['power_w'] for s in selected),'max_memory_mib':max(s['memory_mib'] for s in selected)})
with (a.output/'gpu-samples-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');profiles=['1k','8k','32k','mixed'];x=np.arange(4)
for ax,mode in zip(axes,('replicas','pd')):
    for gpu,color in [(0,'#168aad'),(1,'#a44aab')]:
        vals=[st.mean(r['mean_gpu_util_pct'] for r in rows if r['mode']==mode and r['gpu']==gpu and r['profile']==profile and r['concurrency']==16) for profile in profiles]
        ax.bar(x+(gpu-.5)*.36,vals,.36,color=color,label=f'GPU {gpu}'+(' (prefill)' if mode=='pd' and gpu==0 else ' (decode)' if mode=='pd' else ''))
    ax.set_xticks(x,profiles);ax.set_ylim(0,105);ax.set_ylabel('Mean sampled GPU utilization (%)');ax.set_title('Two replicas' if mode=='replicas' else 'Prefill / decode');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('GPU activity during measured runs · concurrency 16\nOne-second nvidia-smi samples; short runs limit temporal precision',fontsize=13)
for ext in ('png','svg'):
    path=a.output/f'gpu-activity.{ext}';fig.savefig(path,dpi=180,metadata={'Creator':'inference-lab'} if ext=='svg' else {'Software':'inference-lab'})
    if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
