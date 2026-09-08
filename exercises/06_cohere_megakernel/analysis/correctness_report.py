"""Extract and plot the upstream numerical checks without publishing raw logs."""
import argparse
import csv
import json
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('--log',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
text=a.log.read_text()
matches=list(re.finditer(r'^=== bs=(\d+) seqlen=(\d+) (\w+) layer=(\d+) .* (PASS|FAIL) ===$',text,re.M))
rows=[]
for i,m in enumerate(matches):
    section=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
    cosine=[float(x) for x in re.findall(r'cosine=([\d.]+)',section)]
    errors=[float(x) for x in re.findall(r'max_abs=([\d.]+)',section)]
    experts=[float(x) for x in re.findall(r'experts\s+overlap=([\d.]+)',section)]
    rows.append(dict(batch=int(m[1]),context=int(m[2]),layer=m[3],layer_index=int(m[4]),
        min_cosine=min(cosine),max_absolute_error=max(errors),
        expert_overlap=min(experts) if experts else '',passed=m[5]=='PASS'))
assert len(rows)==48 and all(r['passed'] for r in rows)
a.output.mkdir(parents=True,exist_ok=True)
summary={}
for key, filename in [('token_callback','test_token_callback.log'),('kv_bindings','test_kv_bindings.log')]:
    match=re.search(r'^(\d+)/(\d+) passed$',(a.log.parent/filename).read_text(),re.M)
    assert match and match[1]==match[2]=='15'
    summary[key]={'passed':int(match[1]),'total':int(match[2])}
summary['decode_layers']={'cases':len(rows),'passed':sum(r['passed'] for r in rows),
    'tensor_comparisons':len(re.findall(r'cosine=([\d.]+)',text)),
    'min_cosine':min(r['min_cosine'] for r in rows),
    'max_absolute_error':max(r['max_absolute_error'] for r in rows),
    'min_expert_overlap':min(r['expert_overlap'] for r in rows if r['expert_overlap']!=''),
    'batch_sizes':[1,2,4,8],'sequence_lengths':[128,1024,4096,8192],
    'thresholds':{'cosine_min':.99,'max_abs':.5,'expert_overlap_min':.75}}
assert summary['decode_layers']['tensor_comparisons']==128
(a.output/'correctness.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'correctness-cases.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
batches=[1,2,4,8]; contexts=[128,1024,4096,8192]
for ax,layer,title in zip(axes,['dense_full','moe_swa','moe_full'],
    ['Dense + full attention (L0)','MoE + sliding attention (L1)','MoE + full attention (L4)']):
    values=np.array([[next(r['min_cosine'] for r in rows if r['layer']==layer and r['batch']==b and r['context']==c) for c in contexts] for b in batches])
    im=ax.imshow(values,vmin=.99,vmax=1,cmap='YlGnBu',aspect='auto')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{values[i,j]:.5f}',ha='center',va='center',fontsize=9,
                    color='white' if values[i,j]>.997 else '#17252b')
    ax.set_xticks(range(4),['128','1K','4K','8K']); ax.set_yticks(range(4),batches)
    ax.set_xlabel('Prefill context'); ax.set_title(title,fontsize=11)
axes[0].set_ylabel('Batch size')
fig.colorbar(im,ax=axes,label='Minimum tensor cosine similarity',shrink=.85)
fig.suptitle('Actual correctness sweep: 48 / 48 layer cases passed\nCosine ≥ 0.99, max absolute error ≤ 0.50, MoE expert overlap ≥ 0.75',fontsize=13)
for ext in ['png','svg']:
    fig.savefig(a.output/f'correctness-comparison.{ext}',dpi=170)
print('Saved 48 numerical cases and their measured agreement heatmap')

# Normalize Matplotlib path whitespace for clean, reviewable Git diffs.
for name in ['correctness-comparison']:
    svg=a.output/(name+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
