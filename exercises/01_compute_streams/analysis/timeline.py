import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('captures',type=Path,nargs='+');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
fig,axs=plt.subplots(len(args.captures),1,figsize=(13,2.6*len(args.captures)),sharex=True,squeeze=False)
colors=['#22d3ee','#34d399','#a78bfa','#fb923c','#60a5fa']
for ax,p in zip(axs[:,0],args.captures):
 d=json.loads(p.read_text());n=1 if d['strategy']=='batched' else 4
 # First graph's activities, numbered by chronological start (not request identity).
 events=d['events'][:n]
 for i,e in enumerate(events):
  ax.broken_barh([(e['start_us'],e['end_us']-e['start_us'])],(e['lane']-.3,.6),facecolors=colors[e['lane']%len(colors)])
  ax.text((e['start_us']+e['end_us'])/2,e['lane'],str(i),va='center',ha='center',fontsize=9)
 lanes=sorted({e['lane'] for e in events});ax.set_yticks(lanes,[f'Lane {n+1}' for n in lanes]);ax.set_ylim(-.7,max(lanes)+.7);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(f"{d['strategy']} · {d['rows']} rows per request")
axs[-1,0].set_xlabel('Microseconds from first kernel start; common scale')
fig.suptitle('Nsight CUDA Graph node activities · first four-request group\nWSL CUPTI timestamp conversion: reduced accuracy; lanes reflect graph scheduling',fontsize=12)
fig.tight_layout(rect=(0,0,1,.93));args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out.with_suffix('.png'),dpi=170);fig.savefig(args.out.with_suffix('.svg'),metadata={'Date':None})
p=args.out.with_suffix('.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
