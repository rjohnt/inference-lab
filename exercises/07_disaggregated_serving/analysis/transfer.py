"""Plot the isolated NIXL probe's measured descriptor and transfer overhead."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/tuning');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for block,path in [(16,a.raw/'transfer-probe-default.json'),(128,a.raw/'kv128/transfer-probe.json')]:
    d=json.loads(path.read_text());c=d['counters'];n=c['vllm:nixl_xfer_time_seconds_count']
    assert n==4 and c['vllm:nixl_num_failed_transfers_total']==0
    rows.append({'block_tokens':block,'measured_transfers':int(n),'context_tokens':32768,
        'bytes_per_transfer':c['vllm:nixl_bytes_transferred_sum']/n,
        'descriptors_per_transfer':c['vllm:nixl_num_descriptors_sum']/n,
        'mean_transfer_ms':1000*c['vllm:nixl_xfer_time_seconds_sum']/n,
        'mean_post_ms':1000*c['vllm:nixl_post_time_seconds_sum']/n,
        'effective_gb_s':c['vllm:nixl_bytes_transferred_sum']/c['vllm:nixl_xfer_time_seconds_sum']/1e9})
(a.output/'transfer-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained');x=np.arange(2)
post=np.array([r['mean_post_ms'] for r in rows]);total=np.array([r['mean_transfer_ms'] for r in rows]);assert np.all(total>=post)
axes[0].bar(x,post,label='Posting',color='#e59a2a');axes[0].bar(x,total-post,bottom=post,label='Remaining transfer duration',color='#a44aab');axes[0].set_ylabel('Mean NIXL transfer duration (ms)');axes[0].legend(frameon=False,fontsize=9)
axes[1].bar(x,[r['descriptors_per_transfer']/1000 for r in rows],color=['#168aad','#35b4c7']);axes[1].set_ylabel('Descriptors per transfer (thousands)')
for ax in axes:
    ax.set_xticks(x,['16-token blocks','128-token blocks']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Isolated 32K KV transfers · 4.5 GiB per request\nFour measured transfers per configuration, after warmup',fontsize=14)
for ext in ('png','svg'):
    path=a.output/f'transfer-overhead.{ext}';fig.savefig(path,dpi=180,metadata={'Creator':'inference-lab'} if ext=='svg' else {'Software':'inference-lab'})
    if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
print(json.dumps(rows,indent=2))
