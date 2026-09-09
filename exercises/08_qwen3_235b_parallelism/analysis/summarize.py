"""Import reviewed numeric results from private vLLM bench serve captures."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fields=['duration','completed','total_input_tokens','total_output_tokens','request_throughput','output_throughput','mean_ttft_ms','median_ttft_ms','p95_ttft_ms','p99_ttft_ms','mean_tpot_ms','median_tpot_ms','p95_tpot_ms','p99_tpot_ms','mean_itl_ms','p95_itl_ms','p99_itl_ms','mean_e2el_ms','p95_e2el_ms','p99_e2el_ms']
runs=[];requests=[];checks={};validation={};activity=[];samples=[]
for row in csv.reader((a.raw/'gpu-monitor.csv').open()):
    if len(row)!=6:continue
    ts=datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
    samples.append({'time':ts,'gpu':int(row[1]),'util':float(row[2]),'memory_mib':float(row[4]),'power_w':float(row[5])})
for mode in ('tp','pp','ep'):
    folder=a.raw/mode
    if not (folder/'final-metrics.txt').exists():validation[mode]={'status':'incomplete or failed; inspect private logs'};continue
    raw_checks=json.loads((folder/'correctness.json').read_text())
    reference=json.loads((a.raw/'tp/correctness.json').read_text())
    assert len(raw_checks)==len(reference)==6
    checks[mode]=[{'profile':x['profile'],'index':x['index'],'exact_text_match_tp':x['text']==y['text']} for x,y in zip(raw_checks,reference)]
    for meta_path in sorted(folder.glob('*-r[123]-run.json')):
        meta=json.loads(meta_path.read_text());data=json.loads(meta_path.with_name(meta_path.name.replace('-run.json','.json')).read_text())
        assert data['completed']==meta['num_prompts']
        expected={'decode':1024,'intermediate':512,'prefill':128}[meta['profile']]
        assert data['total_output_tokens']==expected*data['completed'],(mode,meta_path.name,data['total_output_tokens'])
        row={k:meta[k] for k in ['mode','profile','concurrency','rep','num_prompts']}
        row.update({k:data[k] for k in fields if k in data and isinstance(data[k],(int,float))});runs.append(row)
        for i,(inp,out,ttft,gaps) in enumerate(zip(data['input_lens'],data['output_lens'],data['ttfts'],data['itls'])):
            requests.append({**{k:row[k] for k in ['mode','profile','concurrency','rep']},'request':i,'input_tokens':inp,'output_tokens':out,'ttft_ms':ttft*1000,'tpot_ms':1000*sum(gaps)/(out-1) if out>1 else 0})
        for gpu in (0,1):
            selected=[s for s in samples if s['gpu']==gpu and meta['started_unix_s']<=s['time']<=meta['ended_unix_s']]
            if selected:activity.append({**{k:row[k] for k in ['mode','profile','concurrency','rep']},'gpu':gpu,'window':'benchmark CLI invocation, including client startup','samples':len(selected),'mean_gpu_util_pct':st.mean(s['util'] for s in selected),'max_memory_mib':max(s['memory_mib'] for s in selected),'mean_power_w':st.mean(s['power_w'] for s in selected)})
    preemptions=0
    for line in (folder/'final-metrics.txt').read_text().splitlines():
        m=re.match(r'vllm:num_preemptions_total(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
        if m:preemptions+=float(m[1])
    validation[mode]={'status':'complete','measured_runs':sum(r['mode']==mode for r in runs),'timed_requests':sum(r['num_prompts'] for r in runs if r['mode']==mode),'preemptions_including_warmup_and_checks':preemptions}
    assert validation[mode]['measured_runs']==27
assert runs,'No completed modes'
summary=[]
for mode in sorted(set(r['mode'] for r in runs)):
    for profile in ('decode','intermediate','prefill'):
        for concurrency in (1,4,16):
            group=[r for r in runs if r['mode']==mode and r['profile']==profile and r['concurrency']==concurrency];assert len(group)==3
            summary.append({'mode':mode,'profile':profile,'concurrency':concurrency,**{key:{'median':st.median(r[key] for r in group),'min':min(r[key] for r in group),'max':max(r[key] for r in group)} for key in fields if all(key in r for r in group)}})
for name,rows in [('runs',runs),('requests',requests),('gpu-activity',activity)]:
    with (a.output/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for name,data in [('summary',summary),('correctness',checks),('validation',validation)]:
    (a.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
manifest=json.loads((a.raw/'dataset-manifest.json').read_text())
for v in manifest['profiles'].values():v.pop('sources',None)
(a.output/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
hardware=json.loads((a.raw/'hardware.json').read_text())
clean={k:v for k,v in hardware.items() if k!='topology'}
link=re.search(r'NV[0-9]+',hardware.get('topology',''))
clean['gpu_interconnect']=link.group() if link else 'see private topology capture'
(a.output/'hardware.json').write_text(json.dumps(clean,indent=2)+'\n')
checkpoint=json.loads((a.raw/'checkpoint.json').read_text());(a.output/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
for axrow,profile in zip(axes,('decode','intermediate','prefill')):
    for ax,key,title,scale in zip(axrow,['output_throughput','mean_ttft_ms','mean_tpot_ms'],['Output tokens/s ↑','Mean TTFT (s) ↓','Mean TPOT (ms) ↓'],[1,1000,1]):
        for mode,color in [('tp','#168aad'),('pp','#e59a2a'),('ep','#a44aab')]:
            rows=sorted([r for r in summary if r['mode']==mode and r['profile']==profile],key=lambda r:r['concurrency'])
            if not rows:continue
            mid=np.array([r[key]['median']/scale for r in rows]);lo=np.array([r[key]['min']/scale for r in rows]);hi=np.array([r[key]['max']/scale for r in rows])
            ax.errorbar([r['concurrency'] for r in rows],mid,yerr=[mid-lo,hi-mid],color=color,marker='o',capsize=3,label=mode.upper())
        ax.set_title(profile+' · '+title,loc='left');ax.set_xticks([1,4,16]);ax.set_xlabel('Client concurrency');ax.grid(alpha=.2)
axes[0,0].legend(frameon=False)
fig.suptitle('Qwen3-235B AWQ · same 2 × A100 80GB\nvLLM bench serve · median and min–max of three measured runs',fontsize=15)
for ext in ('png','svg'):
    path=a.output/f'parallelism.{ext}';fig.savefig(path,dpi=170)
    if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
print(json.dumps(validation,indent=2))
