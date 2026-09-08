import collections, json, statistics, sys
from pathlib import Path

root=Path(sys.argv[1])
metadata=json.loads((root/'metadata.json').read_text())
thinking=metadata.get('args',{}).get('thinking','on')
context=metadata.get('args',{}).get('context',32768)
rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
groups=collections.defaultdict(list)
for r in rows:groups[r['case'],r['mode']].append(r)
def median(rs,key):
    values=[r[key] for r in rs if r.get(key) is not None]
    return statistics.median(values) if values else None
def fmt(x):return '—' if x is None else f'{x:.2f}'
lines=['# Speculative decoding benchmark','',
    'Complete run.' if (root/'COMPLETE').exists() else '**Partial run: do not treat these results as final.**','',
    f'Same Qwen3-8B Q4_K_M target, {context}-token context, q8_0 target KV cache (draft cache settings in command files), GPU layers all, Flash Attention on, thinking {thinking}, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.','',
    'Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.','',
    '| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
summary=[]
for case in sorted(set(r['case'] for r in rows)):
    baseline=groups.get((case,'none'),[])
    b=median(baseline,'first_answer_s')
    for mode in ['none','ngram-mod','draft-eagle3','draft-dflash']:
        rs=groups.get((case,mode),[])
        if not rs:continue
        visible=median(rs,'first_answer_s')
        speed=b/visible if b and visible else None
        rates=[r['timings'].get('predicted_per_second') for r in rs if r['timings'].get('predicted_per_second')]
        rate=statistics.median(rates) if rates else None
        truncated=sum(r['finish_reason']!='stop' for r in rs)
        drafted=sum(r['timings'].get('draft_n',0) for r in rs)
        accepted=sum(r['timings'].get('draft_n_accepted',0) for r in rs)
        acceptance=f'{accepted/drafted:.0%}' if drafted else '—'
        lines.append(f'| {case} | {rs[0]["user_tokens"]} | {mode} | {len(rs)} | {fmt(median(rs,"ttft_s"))} | {fmt(visible)} | {fmt(median(rs,"total_s"))} | {fmt(speed)}× | {fmt(rate)} | {acceptance} | {truncated} |')
        summary.append({'case':case,'mode':mode,'n':len(rs),'visible_speedup':speed,
            'ttft_s':median(rs,'ttft_s'),'first_answer_s':visible,'total_s':median(rs,'total_s'),
            'decode_tps':rate,'draft_acceptance':accepted/drafted if drafted else None,'truncated':truncated})
lines+=['','## Output checks','',
    'Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.','']
for case in sorted(set(r['case'] for r in rows)):
    baseline={r['rep']:r for r in groups.get((case,'none'),[])}
    for mode in ['ngram-mod','draft-eagle3','draft-dflash']:
        pairs=[(r,baseline[r['rep']]) for r in groups.get((case,mode),[]) if r['rep'] in baseline]
        if pairs:lines.append(f'- {case}, {mode}: {sum(a["output_sha256"]==b["output_sha256"] for a,b in pairs)}/{len(pairs)} exact outputs match baseline.')
lines+=['','Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.','',
    'N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.']
(root/'report.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print('\n'.join(lines))
