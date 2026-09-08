"""Combine measured MacBook and GPUStation runs; never synthesize missing cells."""
import ast, collections, hashlib, json, re, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

ROOT=Path(__file__).resolve().parent
GPU=ROOT.parents[1]/'gpustation/dflash2/results-16k'
MAC=ROOT/'results-16k'
OUT=ROOT/'comparison'; OUT.mkdir(exist_ok=True)
configs={}; rows={}
for name,path in [('gpu',GPU),('mac',MAC)]:
    assert (path/'COMPLETE').exists(), f'{name} incomplete'
    configs[name]=json.loads((path/'config.json').read_text())
    rows[name]=[json.loads(s) for s in (path/'results.jsonl').read_text().splitlines()]
    assert len(rows[name])==120 and len({(r['rep'],r['mode'],r['case']) for r in rows[name]})==120
    assert all(r['timings']['cache_n']==0 and r['reasoning_chars']==0 for r in rows[name])
for k in ['context','ubatch','repeats','thinking','draft_max','max_tokens','temperature','seed','cache_prompt','prompts_sha256']:
    assert configs['gpu'][k]==configs['mac'][k], f'{k} differs'
assert configs['gpu']['models']['draft']['sha256']==configs['mac']['models']['draft']['sha256']
cases=json.loads((GPU/'prompts.json').read_text())
assert [(c['id'],c['prompt']) for c in cases]==[(c['id'],c['prompt']) for c in json.loads((MAC/'prompts.json').read_text())]
cols=[('gpu','none','RTX 4070 SUPER\nNo speculation'),('gpu','draft-dflash','RTX 4070 SUPER\nDFlash 2'),('mac','none','MacBook M5 Pro\nNo speculation'),('mac','draft-dflash','MacBook M5 Pro\nDFlash 2')]
summary=[]; checks=[]
expected=dict(owner='team-amber', retries=3, timeout_ms=2750, region='zone-2')
for machine,mode,_ in cols:
    for c in cases:
        rs=[r for r in rows[machine] if r['mode']==mode and r['case']==c['id']]
        assert len(rs)==10
        b={r['rep']:r for r in rows[machine] if r['mode']=='none' and r['case']==c['id']}
        item=dict(machine=machine,mode=mode,case=c['id'],n=10,
            decode_tps=statistics.median(r['timings']['predicted_per_second'] for r in rs),
            total_s=statistics.median(r['total_s'] for r in rs),
            ttft_s=statistics.median(r['ttft_s'] for r in rs),
            exact_matches=sum(r['output_sha256']==b[r['rep']]['output_sha256'] for r in rs),
            output_tokens=statistics.median(r['timings']['predicted_n'] for r in rs))
        summary.append(item)
        for r in rs:
            a=re.sub(r'^```[^\n]*\n|\n```$', '',r['answer'].strip()).strip()
            check=dict(machine=machine,mode=mode,rep=r['rep'],case=c['id'],stopped=r['finish_reason']=='stop',visible=bool(a))
            if c['kind']=='retrieval':
                try: check['retrieval_correct']=json.loads(a)==expected
                except Exception:check['retrieval_correct']=False
            if c['kind']=='code_edit':
                try:ast.parse(a);check['syntax_valid']=True
                except SyntaxError:check['syntax_valid']=False
            checks.append(check)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
lookup={(s['machine'],s['mode'],s['case']):s for s in summary}
fail=sum(c.get('retrieval_correct') is False for c in checks)
trunc=sum(not c['stopped'] for c in checks)
syntax=sum(c.get('syntax_valid') is False for c in checks)
quality=f'240 responses: {fail} retrieval failures; {syntax} Python syntax failures; {trunc} truncated.'
devices=[('mac','MacBook M5 Pro','Q4_K_M target · 64 GB unified memory · macOS / Metal'),
         ('gpu','RTX 4070 Super','IQ2_S target · 12 GB VRAM · Linux / CUDA')]
lines=['# Qwen3.8 27B: two independent DFlash 2 benchmarks','',
'Each benchmark compares No Speculation against DFlash 2 on one fixed device and target model. Target quantizations differ between benchmarks; these results do not establish relative hardware performance.','',
'Both use 16,384 context, the same six prompt texts, 10 repetitions per case/method, thinking off, temperature 0, seed 42, maximum output 1,536 tokens, Q8 KV, batch 512, microbatch 128, one slot, and uncached prompts. DFlash 2 uses seven draft tokens and the same Q4_K_M draft file. Runner revision: 0f3a71be1.','',
'Fresh server per method/repetition, one unmeasured warmup, alternating method order and seeded prompt shuffles. Mac weights are loaded directly by llama.cpp, bypassing the Ollama HTTP scheduler. Its OpenCode configuration remains 64K; this benchmark uses 16K.','']
for machine,title,detail in devices:
    lines += [f'## {title}', '', detail, '', '120 measured responses. Speedup is relative only to this benchmark’s No Speculation baseline.', '',
    '| Case | No Speculation (tok/s) | DFlash 2 (tok/s) | Decode speedup | Baseline request (s) | DFlash request (s) | Exact output agreement |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        b=lookup[machine,'none',c['id']]; d=lookup[machine,'draft-dflash',c['id']]
        lines.append(f"| {c['id']} | {b['decode_tps']:.2f} | {d['decode_tps']:.2f} | {d['decode_tps']/b['decode_tps']:.2f}× | {b['total_s']:.3f} | {d['total_s']:.3f} | {d['exact_matches']}/10 |")
    lines += ['']
lines += [quality, 'Python syntax checks do not establish functional correctness. Decode throughput excludes prefill and startup; request time includes prefill and decoding, excluding startup and warmup.', '',
'Raw Mac run: ../results-16k/; GPU run: ../../../gpustation/dflash2/results-16k/. Configurations, checksums, outputs, logs and timings are retained.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig=plt.figure(figsize=(20,10),facecolor='#f5f7fa')
fig.text(.035,.952,'Qwen3.8 27B | Spec Dec Per Device Benchmark',fontsize=25,fontweight='bold',color='#10243a')
fig.text(.035,.906,'Each panel compares speculation methods on its own fixed model. Different target quantizations; no hardware ranking.',fontsize=14,color='#42556a')
labels=[{'chat':'Short chat','retrieval':'Document','code_edit':'Code edit'}[c['kind']]+f" · {c['user_tokens']:,} tokens" for c in cases]
for group,(machine,title,detail) in enumerate(devices):
    left=.17+group*.50; width=.285
    ax=fig.add_axes([left,.255,width,.48])
    values=np.array([[lookup[machine,mode,c['id']]['decode_tps'] for mode in ['none','draft-dflash']] for c in cases])
    colors=['#e6f0fa','#95bddd','#3972a3','#102f50'] if machine=='mac' else ['#e1f2ec','#93cdb8','#39846c','#164b3d']
    cmap=LinearSegmentedColormap.from_list(machine,colors)
    norm=Normalize(vmin=0,vmax=float(values.max()))
    ax.imshow(values,cmap=cmap,norm=norm,aspect='auto')
    center=.25+group*.50
    fig.text(center,.842,title,ha='center',fontsize=21,fontweight='bold',color='#10243a')
    fig.text(center,.803,detail,ha='center',fontsize=12,color='#42556a')
    ax.set_xticks(range(2),['No Speculation','DFlash 2'],fontsize=13,fontweight='bold')
    ax.xaxis.tick_top()
    ax.set_yticks(range(6),labels,fontsize=11)
    ax.tick_params(axis='both',length=0,pad=12)
    for y in range(6):
        for x in range(2):
            color='white' if norm(values[y,x])>.53 else '#10243a'
            ax.text(x,y-.07,f'{values[y,x]:.1f}',ha='center',va='center',fontsize=25,fontweight='bold',color=color)
            if x==1:
                ax.text(x,y+.25,f'{values[y,x]/values[y,0]:.2f}× baseline',ha='center',va='center',fontsize=11,color=color)
    ax.set_xticks(np.arange(-.5,2,1),minor=True)
    ax.set_yticks(np.arange(-.5,6,1),minor=True)
    ax.grid(which='minor',color='#f5f7fa',linewidth=5)
    ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(center,.210,'Median decode tok/s · 10 runs per cell · 120 responses',ha='center',fontsize=12,color='#42556a')
    fig.text(center,.176,'Color intensity scaled independently within this panel',ha='center',fontsize=11,color='#607084')
fig.add_artist(plt.Line2D([.5,.5],[.17,.86],transform=fig.transFigure,color='#c6ced7',linewidth=1.5))
fig.text(.035,.115,'Protocol: 16K context · identical prompts · thinking off · Q8 KV · uncached requests · 7-token DFlash 2 blocks',fontsize=12,color='#42556a')
fig.text(.035,.077,'Decode excludes prefill and startup. Multipliers compare DFlash 2 only with the baseline in the same panel.',fontsize=12,color='#42556a')
fig.text(.035,.038,quality+' See report for request times and output agreement.',fontsize=11,color='#42556a')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'throughput-comparison.{ext}',dpi=180,facecolor=fig.get_facecolor())
print('\n'.join(lines))
