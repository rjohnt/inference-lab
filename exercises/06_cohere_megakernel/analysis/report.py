"""Validate raw runs and publish only compact, non-sensitive performance data."""
import argparse
import csv
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser()
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for context in (1024,8192,32768):
    for batch in (1,2,4,8):
        for rep in (1,2,3):
            for engine,prefix in [('megakernel','mk'),('vllm','vllm')]:
                path=a.raw/f'{prefix}-c{context}-b{batch}-r{rep}.json'
                d=json.loads(path.read_text())
                if engine=='megakernel':
                    assert d['decode_steps']==511
                    assert d['output_tokens_per_request']==512
                    tpot=d['tpot_ms']
                else:
                    assert d['completed']==batch, path
                    assert d['total_output_tokens']==batch*512, path
                    assert d['output_lens']==[512]*batch, path
                    assert d['input_lens']==[context]*batch, path
                    assert not any(d.get('errors',[])), path
                    tpot=d['mean_tpot_ms']
                assert tpot>0
                rows.append(dict(engine=engine,context=context,batch=batch,rep=rep,
                    tpot_ms=tpot,decode_tok_s=batch*1000/tpot))
with (a.output/'decode-runs.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
summary=[]
for context in (1024,8192,32768):
    for batch in (1,2,4,8):
        r={'context':context,'batch':batch}
        for engine in ('megakernel','vllm'):
            vals=[x['tpot_ms'] for x in rows if x['context']==context and x['batch']==batch and x['engine']==engine]
            r[f'{engine}_tpot_ms']=statistics.median(vals)
            r[f'{engine}_min_ms']=min(vals)
            r[f'{engine}_max_ms']=max(vals)
        r['speedup']=r['vllm_tpot_ms']/r['megakernel_tpot_ms']
        summary.append(r)
(a.output/'decode-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| Context | Batch | Megakernel ms/token | vLLM ms/token | Speedup |',
       '|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['context']:,} | {r['batch']} | {r['megakernel_tpot_ms']:.3f} | {r['vllm_tpot_ms']:.3f} | {r['speedup']:.2f}× |")
(a.output/'decode-table.md').write_text('\n'.join(lines)+'\n')

api=[]
requests=[]
for engine in ('megakernel','vllm'):
    results=json.loads((a.raw/f'api-{engine}.json').read_text())
    assert len(results)==6
    for r in results:
        assert len(r['requests'])==8
        assert all(0<x['completion_tokens']<=256 and x['finish_reason'] in ('stop','length') for x in r['requests'])
        row={k:r[k] for k in ('engine','concurrency','rep','wall_s','output_tok_s','mean_ttft_ms','mean_tpot_ms')}
        row['output_tokens']=sum(x['completion_tokens'] for x in r['requests'])
        row['prompt_tokens']=sum(x['prompt_tokens'] for x in r['requests'])
        api.append(row)
        for index, request in enumerate(r['requests']):
            requests.append(dict(engine=engine,concurrency=r['concurrency'],rep=r['rep'],task_index=index,
                **{k:request[k] for k in ('prompt_tokens','completion_tokens','ttft_ms','tpot_ms','latency_ms','finish_reason')}))
with (a.output/'serving-runs.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(api[0]),lineterminator="\n"); w.writeheader(); w.writerows(api)
with (a.output/'serving-requests.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(requests[0]),lineterminator="\n"); w.writeheader(); w.writerows(requests)
for concurrency in (1,8):
    for rep in (1,2,3):
        for index in range(8):
            pair=[x for x in requests if x['concurrency']==concurrency and x['rep']==rep and x['task_index']==index]
            assert len(pair)==2 and pair[0]['prompt_tokens']==pair[1]['prompt_tokens'], pair
serving=[]
for concurrency in (1,8):
    r={'concurrency':concurrency}
    for engine in ('megakernel','vllm'):
        group=[x for x in api if x['engine']==engine and x['concurrency']==concurrency]
        assert len(group)==3
        for metric in ('output_tok_s','mean_ttft_ms','mean_tpot_ms'):
            r[f'{engine}_{metric}']=statistics.median(x[metric] for x in group)
            r[f'{engine}_{metric}_min']=min(x[metric] for x in group)
            r[f'{engine}_{metric}_max']=max(x[metric] for x in group)
        r[f'{engine}_output_tokens']=sum(x['output_tokens'] for x in group)
    r['throughput_speedup']=r['megakernel_output_tok_s']/r['vllm_output_tok_s']
    serving.append(r)
(a.output/'serving-summary.json').write_text(json.dumps(serving,indent=2)+'\n')
lines=['| Concurrency | Megakernel tok/s | vLLM tok/s | Throughput speedup |',
       '|---:|---:|---:|---:|']
for r in serving:
    lines.append(f"| {r['concurrency']} | {r['megakernel_output_tok_s']:.1f} | {r['vllm_output_tok_s']:.1f} | {r['throughput_speedup']:.2f}× |")
lines+=['','| Concurrency | MK TTFT ms | vLLM TTFT ms | MK TPOT ms | vLLM TPOT ms |',
        '|---:|---:|---:|---:|---:|']
for r in serving:
    lines.append(f"| {r['concurrency']} | {r['megakernel_mean_ttft_ms']:.1f} | {r['vllm_mean_ttft_ms']:.1f} | {r['megakernel_mean_tpot_ms']:.2f} | {r['vllm_mean_tpot_ms']:.2f} |")
(a.output/'serving-table.md').write_text('\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,context in zip(axes,(1024,8192,32768)):
    group=[r for r in summary if r['context']==context]
    for engine,label,color in [('megakernel','Megakernel','#25856b'),('vllm','vLLM 0.24.0','#6977c6')]:
        vals=[r[f'{engine}_tpot_ms'] for r in group]
        err=[[v-r[f'{engine}_min_ms'] for v,r in zip(vals,group)],
             [r[f'{engine}_max_ms']-v for v,r in zip(vals,group)]]
        ax.errorbar([1,2,4,8],vals,yerr=err,marker='o',capsize=3,label=label,color=color)
    ax.set_title(f'{context:,} context tokens')
    ax.set_xlabel('Batch size'); ax.set_xticks([1,2,4,8]); ax.grid(alpha=.2)
axes[0].set_ylabel('Decode ms / output token (lower is better)')
axes[-1].legend()
fig.suptitle('North Mini Code BF16 · one H100 SXM · median and range of 3 runs')
fig.text(.5,.01,'Timing scopes: megakernel native loop vs vLLM HTTP TPOT after TTFT; not a pure kernel comparison.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,1))
for ext in ('png','svg'):
    fig.savefig(a.output/f'decode-comparison.{ext}',dpi=170)
fig,axes=plt.subplots(1,2,figsize=(9,3.8))
for ax,metric,label in zip(axes,('output_tok_s','mean_ttft_ms'),
    ('Total output tokens / second (higher is better)','Mean TTFT, milliseconds (lower is better)')):
    for offset,engine,name,color in [(-.18,'megakernel','Megakernel','#25856b'),(.18,'vllm','vLLM 0.24.0','#6977c6')]:
        values=[r[f'{engine}_{metric}'] for r in serving]
        errors=[[v-r[f'{engine}_{metric}_min'] for v,r in zip(values,serving)],
                [r[f'{engine}_{metric}_max']-v for v,r in zip(values,serving)]]
        ax.bar([i+offset for i in range(2)],values,width=.35,label=name,color=color,yerr=errors,capsize=3)
    ax.set_xticks([0,1],['Concurrency 1','Concurrency 8'])
    ax.set_ylabel(label); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
axes[0].legend()
fig.suptitle('Same streaming API client · eight coding requests · median + range of 3 runs')
fig.tight_layout()
for ext in ('png','svg'):
    fig.savefig(a.output/f'serving-comparison.{ext}',dpi=170)
print(f'Validated {len(rows)} decode runs and {len(api)} serving runs')

# Normalize Matplotlib path whitespace for clean, reviewable Git diffs.
for name in ['decode-comparison', 'serving-comparison']:
    svg=a.output/(name+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
