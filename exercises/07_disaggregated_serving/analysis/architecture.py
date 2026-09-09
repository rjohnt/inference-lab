"""Draw the two serving configurations executed by src/run.py."""
from pathlib import Path
from html import escape
p=Path(__file__).resolve().parents[1]/'results/architecture.svg'
s=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 700">
<defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern>
<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#94a3b8"/></marker>
<marker id="kv" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#a78bfa"/></marker>
</defs><style>text {font-family: 'SF Mono', 'DejaVu Sans Mono', monospace;}</style>
<rect width="1080" height="700" fill="#0f172a"/><rect width="1080" height="700" fill="url(#grid)"/>
''']
def text(x,y,t,size=12,color='#e2e8f0',anchor='middle'):
    s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(t)}</text>')
def box(x,y,w,title,sub,color):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="65" rx="8" fill="#0f172a"/>')
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="65" rx="8" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-width="1.5"/>')
    text(x+w/2,y+27,title,14);text(x+w/2,y+48,sub,10,'#94a3b8')
def arrow(d,color='#94a3b8',marker='arrow'):
    s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#{marker})"/>')
text(30,35,'Qwen3-8B: disaggregation versus two replicas',22,anchor='start')
text(30,60,'Same 2 × A100-SXM4-80GB • BF16 • TP=1 per worker • vLLM 0.24.0',13,'#94a3b8','start')
for x,label in [(30,'BASELINE · TWO FULL-SERVING REPLICAS'),(555,'EXPERIMENT · PREFILL / DECODE')]:
    s.append(f'<rect x="{x}" y="95" width="495" height="490" rx="12" fill="none" stroke="#fbbf24" stroke-dasharray="8 4"/>')
    text(x+20,122,label,13,'#fbbf24','start')
arrow('M 277 215 V 258');arrow('M 277 323 V 368 H 160 V 415');arrow('M 277 368 H 390 V 415')
arrow('M 802 215 V 258');arrow('M 730 323 V 365 H 680 V 415');arrow('M 874 323 V 365 H 925 V 415')
arrow('M 775 450 H 830','#a78bfa','kv')
box(152,150,250,'Streaming benchmark client','1K / 8K / 32K / mixed prompts','#22d3ee')
box(152,258,250,'Least-outstanding router','Each request stays on one replica','#fb923c')
box(65,415,190,'GPU 0 · replica','Full prefill + full decode','#34d399')
box(295,415,190,'GPU 1 · replica','Full prefill + full decode','#34d399')
box(677,150,250,'Same benchmark client','Concurrency 4 / 16; 128 tokens','#22d3ee')
box(677,258,250,'P/D request router','Prefill response → decode request','#fb923c')
box(585,415,190,'GPU 0 · prefill','Full Qwen3-8B model','#34d399')
box(830,415,190,'GPU 1 · decode','Full Qwen3-8B model','#a78bfa')
text(680,390,'1. Prompt',11,'#94a3b8');text(930,390,'2. Prompt + metadata',11,'#94a3b8')
text(802,505,'NIXL / UCX pulls GPU KV cache',13,'#a78bfa')
text(802,530,'Peer topology: NV12',12,'#94a3b8')
text(277,525,'No inter-worker KV transfer',13,'#94a3b8')
text(277,550,'Each GPU holds the full model',12,'#94a3b8')
text(30,624,'Validation: greedy output comparison + required transfer metadata + NIXL counters',13,anchor='start')
text(30,650,'Timing: client TTFT, time/output token, chunk gaps, aggregate output tokens/s',13,'#94a3b8','start')
text(30,676,'PyTorch peer-copy bandwidth is a separate control; it is not NIXL transfer bandwidth.',11,'#94a3b8','start')
s.append('</svg>');p.write_text('\n'.join(s)+'\n')
