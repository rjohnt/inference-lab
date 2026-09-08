"""Plot measured SM ranges; retain only reviewed aggregates, not the raw trace."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument('--raw', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
events = json.loads((a.raw / 'mk-sm-trace.json').read_text())['traceEvents']
events = [e for e in events if e['ph'] == 'X']
assert events and all(e['dur'] >= 0 for e in events)
sms = sorted({e['args']['sm_id'] for e in events})
assert len(sms) == 132, len(sms)
start = min(e['ts'] for e in events)
end = max(e['ts'] + e['dur'] for e in events)
colors = {'QKV / output projection': '#22d3ee', 'Attention': '#60a5fa',
          'MoE routing / gather': '#fbbf24', 'MoE GEMMs': '#34d399',
          'Combine / normalization': '#a78bfa', 'Dense FFN': '#fb923c',
          'LM head': '#fb7185', 'Other': '#94a3b8'}

def group(name):
    if name in ('qkv', 'oproj'): return 'QKV / output projection'
    if name.startswith('attn'): return 'Attention'
    if name.startswith('route') or name == 'moe_gather': return 'MoE routing / gather'
    if name in ('moe_upgate_act', 'moe_down'): return 'MoE GEMMs'
    if name in ('moe_combine', 'rmsnorm'): return 'Combine / normalization'
    if name.startswith('ffn'): return 'Dense FFN'
    if name == 'lm_head': return 'LM head'
    return 'Other'

spans_by_lane = defaultdict(list)
for e in events:
    spans_by_lane[e['args']['sm_id'], group(e['name'])].append((e['ts']-start, e['dur']))

with plt.rc_context({'figure.facecolor': '#0f172a', 'axes.facecolor': '#0f172a',
                     'text.color': '#e2e8f0', 'axes.labelcolor': '#cbd5e1',
                     'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
                     'axes.edgecolor': '#475569', 'font.size': 10}):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [1.4, 1]})
    for ax, limit, lanes in [(axes[0], end-start, sms), (axes[1], 200, sms[:32])]:
        for sm in lanes:
            for category, color in colors.items():
                spans = [span for span in spans_by_lane[sm, category] if span[0] < limit]
                if spans:
                    ax.broken_barh(spans, (sm-.43, .86), facecolors=color,
                                   edgecolors='none', rasterized=True)
        ax.set_xlim(0, limit)
        ax.set_ylim(lanes[-1]+1, lanes[0]-1)
        ax.set_ylabel('SM ID')
        ax.set_xlabel('Microseconds from first recorded event')
        ax.grid(axis='x', alpha=.12)
    axes[0].set_title(f'One actual instrumented decode step · {len(events):,} ranges across {len(sms)} SMs', loc='left')
    axes[1].set_title(f'Detail: first 200 µs, first 32 recorded SMs (IDs {sms[0]}–{sms[31]})', loc='left')
    present = {group(e['name']) for e in events}
    fig.legend(handles=[Patch(color=v, label=k) for k, v in colors.items() if k in present],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .04))
    fig.suptitle('North Mini Code BF16 · H100 SXM · batch 1 · synthetic 8K KV', fontsize=15)
    fig.text(.5, .015, 'Diagnostic Python path; instrumentation changes timing. Blank space is not an occupancy measurement.',
             ha='center', color='#94a3b8', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .95))
    a.output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(a.output / f'sm-timeline.{ext}', dpi=180, metadata={'Creator': 'inference-lab'})

comparison = re.search(r'non_profile_cuda=([\d.]+) ms profiled_cuda=([\d.]+) ms trace_span=([\d.]+) ms.*instrumentation_overhead=([+\-\d.]+)%',
                       (a.raw / 'profile.log').read_text())
assert comparison, 'Missing instrumented/uninstrumented timing comparison'
summary = {'batch': 1, 'context': 8192, 'sms': len(sms), 'event_count': len(events),
           'trace_span_us': end-start, 'event_counts_by_operation': dict(sorted(Counter(e['name'] for e in events).items())),
           **dict(zip(('non_profile_cuda_ms', 'profiled_cuda_ms', 'reported_trace_span_ms', 'instrumentation_overhead_percent'),
                      map(float, comparison.groups())))}
(a.output / 'profile-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))

# Normalize Matplotlib path whitespace for clean, reviewable Git diffs.
for name in ['sm-timeline']:
    svg=a.output/(name+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
