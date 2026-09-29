"""Render the reviewed timings in notes/baseline.md; no new benchmark runs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = [('cold: ping',13.35,16.25,18.44),('warm: ping',.29,2.18,2.41),('warm: diffusion model',.48,9.32,18.90),('warm: model identity',3.49,4.72,5.28)]
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e4ede9','axes.labelcolor':'#94aaa1','xtick.color':'#94aaa1','ytick.color':'#e4ede9','svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(10,6),facecolor='#101b16')
ax.set_facecolor('#101b16')
y=np.arange(len(rows)); left=np.zeros(4)
for label,color,values in [('until first token','#34d399',[r[1] for r in rows]),('until first answer','#a78bfa',[r[2]-r[1] for r in rows]),('remaining response','#456b5c',[r[3]-r[2] for r in rows])]:
 ax.barh(y,values,left=left,height=.48,label=label,color=color);left+=values
for i,r in enumerate(rows):ax.text(r[3]+.3,i,f'{r[3]:.2f} s',va='center',fontsize=11)
ax.set_yticks(y,[r[0] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,22);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('elapsed time (seconds)',labelpad=12)
ax.grid(axis='x',color='#345044',alpha=.5);ax.set_axisbelow(True)
for spine in ax.spines.values():spine.set_visible(False)
ax.tick_params(axis='y',length=0,pad=12)
fig.text(.06,.94,'Ollama streaming baseline',fontsize=22,weight='bold')
fig.text(.06,.885,'Qwen3 8B · 32K context · thinking enabled · client-side timings',fontsize=11,color='#94aaa1')
ax.legend(loc='upper left',bbox_to_anchor=(0,-.23),ncol=3,frameon=False,labelcolor='#e4ede9',fontsize=10)
fig.text(.06,.065,'Cold: one sample. Warm: median of three runs per prompt. Prompt cache may be reused.',fontsize=9,color='#94aaa1')
fig.text(.06,.03,'First token includes reasoning; first answer is the first answer-content delta. Historical observations.',fontsize=9,color='#94aaa1')
fig.subplots_adjust(left=.26,right=.94,top=.81,bottom=.28)
out=Path(__file__).resolve().parents[1]/'results';out.mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(out/f'streaming-timings.{ext}',dpi=180,facecolor=fig.get_facecolor())
