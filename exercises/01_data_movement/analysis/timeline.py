"""Plot CUDA-event stage spans from a separate instrumented run."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('captures',nargs='+',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
sources=[json.loads(p.read_text()).get('source','CUDA events') for p in args.captures]
hardware=all(s=='Nsight Systems CUDA activity trace' for s in sources)
fig,axs=plt.subplots(len(args.captures),1,figsize=(14,3.5*len(args.captures)),squeeze=False,sharex=True)
for ax,path in zip(axs[:,0],args.captures):
 data=json.loads(path.read_text())
 for e in data['events']:
  y=['H2D','compute','D2H'].index(e['stage']);color=['#22d3ee','#34d399','#a78bfa'][y]
  if e['batch']<9:
   ax.broken_barh([(e['start_ms'],e['end_ms']-e['start_ms'])],(y-.35,.7),facecolors=color)
   ax.text((e['start_ms']+e['end_ms'])/2,y,str(e['batch']),ha='center',va='center',fontsize=8)
 ax.set_yticks(range(3),['Upload','Compute','Download']);ax.invert_yaxis();ax.set_xlabel('Milliseconds from first GPU activity' if hardware else 'Milliseconds from CUDA-event origin');ax.set_title(data['workload']+' · '+data['variant']);ax.grid(axis='x',alpha=.2)
fig.suptitle('Nsight Systems GPU activities · first 9 batches · separate profiled capture\nWSL CUPTI timestamp conversion: reduced timestamp accuracy' if hardware else 'Measured CUDA-event spans · first 9 batches · separate instrumented capture\nSpans may include GPU scheduling gaps; this is not a hardware-engine trace',fontsize=12)
fig.tight_layout(rect=(0,0,1,.93));args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out.with_suffix('.png'),dpi=170);fig.savefig(args.out.with_suffix('.svg'),metadata={'Date':None})
p=args.out.with_suffix('.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
