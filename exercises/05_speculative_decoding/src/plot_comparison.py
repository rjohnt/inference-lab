"""Render measured benchmark values locally; no generated or interpolated results."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
root=Path(sys.argv[1])
assert (root/'COMPLETE').exists(), 'Only plot a complete run'
summary=json.loads((root/'summary.json').read_text())
meta=json.loads((root/'metadata.json').read_text())
thinking=meta['args'].get('thinking','on')
cases=['chat_short','document_512','document_2048','document_8192','code_512','code_8192']
labels=['Short chat · 12 tokens','Document · 578 tokens','Document · 2,115 tokens','Document · 8,258 tokens','Code edit · 634 tokens','Code edit · 8,314 tokens']
modes=['none','draft-eagle3','draft-dflash']
groups={(r['case'],r['mode']):r for r in summary}
repeats=meta['args']['repeats']
assert all(groups[c,m]['n']==repeats for c in cases for m in modes)
data=np.array([[groups[c,m]['decode_tps'] for m in modes] for c in cases])
ratios=data/data[:,0:1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig=plt.figure(figsize=(12,8),facecolor='#f5f7fa')
fig.text(.06,.945,'Qwen3-8B | Speculative decoding',fontsize=23,fontweight='bold',color='#10243a')
fig.text(.06,.895,'Decode throughput · tokens / second · higher values are darker',fontsize=13,color='#42556a')
ax=fig.add_axes([.32,.22,.60,.59])
cmap=LinearSegmentedColormap.from_list('benchmark',['#e6f0fa','#95bddd','#3972a3','#102f50'])
norm=Normalize(vmin=float(data.min())*.85,vmax=float(data.max()))
ax.imshow(data,cmap=cmap,norm=norm,aspect='auto')
ax.set_xticks(range(3),['No speculation','EAGLE-3','DFlash 1*' if thinking=='on' else 'DFlash 1'],fontsize=14,fontweight='bold')
ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=14)
ax.set_yticks(range(6),labels,fontsize=12)
for y in range(6):
    for x in range(3):
        color='white' if norm(data[y,x])>.53 else '#10243a'
        ax.text(x,y-.10,f'{data[y,x]:.1f}',ha='center',va='center',fontsize=22,fontweight='bold',color=color)
        ax.text(x,y+.23,'baseline' if x==0 else f'{ratios[y,x]:.2f}× baseline',ha='center',va='center',fontsize=10,color=color)
ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
ax.grid(which='minor',color='#f5f7fa',linewidth=5);ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.06,.14,f"RTX 4070 SUPER 12 GB  •  {meta['args']['context']:,}-token context  •  Q4_K_M target / Q8 KV",fontsize=11,color='#42556a')
token_label = 'includes reasoning tokens' if thinking == 'on' else 'visible answer tokens'
fig.text(.06,.105,f'Median of {repeats} runs per cell · {repeats*18} requests · thinking {thinking} · {token_label}',fontsize=11,color='#42556a')
fig.text(.06,.070,'Warm server-local decoding; excludes prompt prefill and network latency. Speedup is throughput ratio.',fontsize=10,color='#607084')
if thinking=='on':fig.text(.06,.025,'* DFlash draft is documented for non-thinking use; these runs enabled thinking. Retest needed.',fontsize=10,color='#9b3528')
for ext in ['png','svg','pdf']:
    fig.savefig(root/f'throughput-comparison.{ext}',dpi=200,facecolor=fig.get_facecolor())
print(root/'throughput-comparison.png')
