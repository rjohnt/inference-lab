"""Validate coverage and render measured values from a completed matched run."""
import ast
import collections
import json
from pathlib import Path
import re
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

root = Path(sys.argv[1])
assert (root / 'COMPLETE').exists(), 'Run is incomplete'
config = json.loads((root / 'config.json').read_text())
rows = [json.loads(line) for line in (root / 'results.jsonl').read_text().splitlines()]
cases = json.loads((root / 'prompts.json').read_text())
modes = ['none', 'draft-dflash']
groups = collections.defaultdict(list)
for row in rows:
    groups[row['case'], row['mode']].append(row)
n = config['repeats']
assert len(rows) == len(cases) * len(modes) * n
assert len({(r['case'], r['mode'], r['rep']) for r in rows}) == len(rows)
assert all(len(groups[c['id'],m]) == n for c in cases for m in modes)
assert all(r['timings'].get('cache_n') == 0 for r in rows)

reference = ast.parse('''def summarize(values):
    if not values:
        return {"total": 0, "count": 0, "average": None, "minimum": None, "maximum": None}
    total = sum(values)
    count = len(values)
    average = total / count
    minimum = min(values)
    maximum = max(values)
    return {"total": total, "count": count, "average": average, "minimum": minimum, "maximum": maximum}
''')
expected = dict(owner='team-amber', retries=3, timeout_ms=2750, region='zone-2')
checks = []
for r in rows:
    answer = r['answer'].strip()
    if answer.startswith('```'):
        answer = re.sub(r'^```[^\n]*\n|\n```$', '', answer).strip()
    check = dict(case=r['case'], mode=r['mode'], rep=r['rep'], stopped=r['finish_reason']=='stop',
                 visible_answer=bool(r['answer'].strip()), reasoning_chars=r['reasoning_chars'])
    if r['case'].startswith('document'):
        try:
            check['retrieval_correct'] = json.loads(answer) == expected
        except Exception:
            check['retrieval_correct'] = False
    if r['case'].startswith('code'):
        try:
            tree = ast.parse(answer)
            check['matches_reference_ast'] = ast.dump(tree) == ast.dump(reference)
            check['syntax_valid'] = True
        except SyntaxError:
            check['syntax_valid'] = False
            check['matches_reference_ast'] = False
    checks.append(check)
(root / 'validation.json').write_text(json.dumps(checks, indent=2))
summary = []
for c in cases:
    baseline = {r['rep']: r for r in groups[c['id'], 'none']}
    for mode in modes:
        rs = groups[c['id'], mode]
        drafted = sum(r['timings'].get('draft_n', 0) for r in rs)
        accepted = sum(r['timings'].get('draft_n_accepted', 0) for r in rs)
        summary.append(dict(case=c['id'], mode=mode, n=len(rs),
            decode_tps=statistics.median(r['timings']['predicted_per_second'] for r in rs),
            total_s=statistics.median(r['total_s'] for r in rs),
            ttft_s=statistics.median(r['ttft_s'] for r in rs if r['ttft_s'] is not None),
            draft_acceptance=accepted/drafted if drafted else None,
            exact_output_matches=sum(r['output_sha256']==baseline[r['rep']]['output_sha256'] for r in rs),
            truncated=sum(r['finish_reason'] != 'stop' for r in rs)))
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
by = {(r['case'],r['mode']):r for r in summary}
data = np.array([[by[c['id'], m]['decode_tps'] for m in modes] for c in cases])
ratios = data/data[:, :1]
retrieval_bad = sum(c.get('retrieval_correct') is False for c in checks)
code_review = sum(c.get('matches_reference_ast') is False for c in checks)
truncated = sum(not c['stopped'] for c in checks)
quality = f'Quality: {retrieval_bad} retrieval failures; {code_review} code outputs need reference review; {truncated} truncated.'
lines = ['# Qwen3.8-27B UD-IQ2_S: baseline versus DFlash 2', '',
         f'{len(rows)} measured requests; {n} repetitions per cell; thinking off; context {config["context"]:,}.', '',
         'Exact same six prompt texts as the previous Qwen3-8B run, retokenized for this target. Method order alternates by repetition. Warm, uncached requests; greedy sampling, seed 42, 1536-token output cap; Q8 KV; all model layers requested on GPU. Raw logs record actual placement.', '',
         quality, 'Code review means the AST differs from the reference; this alone does not establish functional incorrectness.', '',
         '| Case | Baseline tok/s | DFlash 2 tok/s | Decode ratio | Baseline total s | DFlash total s | Total-time speedup | Exact matches |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for c in cases:
    b,d = [by[c['id'], m] for m in modes]
    lines.append(f'| {c["id"]} | {b["decode_tps"]:.2f} | {d["decode_tps"]:.2f} | {d["decode_tps"]/b["decode_tps"]:.2f}× | {b["total_s"]:.3f} | {d["total_s"]:.3f} | {b["total_s"]/d["total_s"]:.2f}× | {d["exact_output_matches"]}/{n} |')
lines += ['', 'Decode throughput excludes prompt prefill; total time includes it. This measures the IQ2_S target and Q4_K_M drafter on this PC; it is not a direct algorithm comparison with the earlier Qwen3-8B DFlash 1 results.', '',
          'Model revisions, SHA256 checksums, server version and settings: config.json. Per-request output and metrics: results.jsonl. Quality checks: validation.json.']
(root / 'report.md').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12})
fig = plt.figure(figsize=(12,8), facecolor='#f5f7fa')
fig.text(.06,.945,'Qwen3.8-27B UD-IQ2_S | DFlash 2', fontsize=23,fontweight='bold',color='#10243a')
fig.text(.06,.895,'Decode throughput · tokens / second · higher values are darker',fontsize=13,color='#42556a')
ax = fig.add_axes([.36,.24,.56,.57])
cmap = LinearSegmentedColormap.from_list('benchmark',['#e6f0fa','#95bddd','#3972a3','#102f50'])
norm = Normalize(vmin=float(data.min())*.85, vmax=float(data.max()))
ax.imshow(data,cmap=cmap,norm=norm,aspect='auto')
ax.set_xticks(range(2), ['No speculation','DFlash 2'],fontsize=14,fontweight='bold')
ax.xaxis.tick_top()
ax.tick_params(axis='both',length=0,pad=14)
labels = [('Short chat' if c['kind']=='chat' else 'Document' if c['kind']=='retrieval' else 'Code edit') + f' · {c["user_tokens"]:,} tokens' for c in cases]
ax.set_yticks(range(len(cases)),labels,fontsize=12)
for y in range(len(cases)):
    for x in range(2):
        color='white' if norm(data[y,x])>.53 else '#10243a'
        ax.text(x,y-.10,f'{data[y,x]:.1f}',ha='center',va='center',fontsize=24,fontweight='bold',color=color)
        ax.text(x,y+.23,'baseline' if x==0 else f'{ratios[y,x]:.2f}× baseline',ha='center',va='center',fontsize=11,color=color)
ax.set_xticks(np.arange(-.5,2,1),minor=True)
ax.set_yticks(np.arange(-.5,len(cases),1),minor=True)
ax.grid(which='minor',color='#f5f7fa',linewidth=5)
ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(.06,.16,f'RTX 4070 SUPER 12 GB · {config["context"]:,}-token context · IQ2_S target / Q4_K_M draft / Q8 KV',fontsize=11,color='#42556a')
fig.text(.06,.12,f'Median of {n} runs per cell · {len(rows)} requests · thinking off · identical prompts for both methods',fontsize=11,color='#42556a')
fig.text(.06,.08,'Warm server-local decoding; excludes prompt prefill and network latency. Speedup is throughput ratio.',fontsize=10,color='#607084')
fig.text(.06,.04,quality,fontsize=10,color='#9b3528' if retrieval_bad or code_review or truncated else '#607084')
for extension in ['png','svg','pdf']:
    fig.savefig(root / f'throughput-comparison.{extension}',dpi=200,facecolor=fig.get_facecolor())
print('\n'.join(lines))
print(root / 'throughput-comparison.png')
