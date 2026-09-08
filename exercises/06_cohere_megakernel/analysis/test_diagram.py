"""Draw the completed experiment from validated, measured result counts."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

base=Path(__file__).resolve().parents[1]
correct=json.loads((base/'results/correctness.json').read_text())
decode=list(csv.DictReader((base/'results/decode-runs.csv').open()))
api=list(csv.DictReader((base/'results/serving-runs.csv').open()))
assert len(decode)==72 and len(api)==12
assert correct['decode_layers']['passed']==48

parts=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 760">
<style>text{font-family:'SF Mono','Cascadia Code',monospace} .title{font-size:22px;font-weight:700;fill:#f8fafc}.name{font-size:16px;font-weight:600;fill:#f8fafc}.body{font-size:13px;fill:#cbd5e1}.note{font-size:11px;fill:#94a3b8}</style>
<defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#64748b"/></marker></defs>
<rect width="1080" height="760" fill="#0f172a"/><rect width="1080" height="760" fill="url(#grid)"/>
<rect x="30" y="95" width="1020" height="600" rx="14" fill="none" stroke="#fbbf24" stroke-dasharray="8 4"/>
<g fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)">
<path d="M540 200V245"/>
<path d="M540 325V355H285V390"/><path d="M540 325V355H795V390"/>
<path d="M285 510V545H540V575"/><path d="M795 510V545H540V575"/>
</g>''']

def box(x,y,w,h,color,fill,name,lines):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#0f172a"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" fill-opacity=".4" stroke="{color}" stroke-width="1.5"/>')
    parts.append(f'<text x="{x+w/2}" y="{y+27}" text-anchor="middle" class="name">{escape(name)}</text>')
    for i,line in enumerate(lines):
        parts.append(f'<text x="{x+w/2}" y="{y+49+i*19}" text-anchor="middle" class="body">{escape(line)}</text>')

box(330,130,420,70,'#22d3ee','#083344','One pinned North Mini Code checkpoint',
    ['30.48B parameters · BF16 · same weights'])
box(330,245,420,80,'#34d399','#064e3b','Correctness verified before timing',
    ['15/15 callback · 15/15 KV tests',f"{correct['decode_layers']['passed']}/48 layer cases · 128 tensor comparisons"])
box(65,390,440,120,'#34d399','#064e3b','Decode-only comparison',
    ['Megakernel native loop vs vLLM after TTFT','1K / 8K / 32K · batches 1 / 2 / 4 / 8',
     f'{len(decode)} measured runs · 512 tokens/request','Real weights + synthetic KV; prefill excluded'])
box(575,390,440,120,'#60a5fa','#1e3a8a','Same streaming API client',
    ['Megakernel server vs vLLM 0.24.0','8 coding tasks · concurrency 1 and 8',
     f'{len(api)} measured sets · 48 requests/engine','Prefill, scheduling and HTTP included'])
box(330,575,420,80,'#a78bfa','#4c1d95','Reviewed evidence in inference-lab',
    ['Median + range from three repetitions','CSV measurements · PNG charts · SVG diagrams'])
parts+=['<text x="30" y="38" class="title">What we actually ran</text>',
    '<text x="30" y="65" class="body">Cohere megakernel reproduction · completed test topology</text>',
    '<text x="50" y="118" class="note" fill="#fbbf24">ONE H100 SXM 80 GB · 132 SMs · ENGINES RUN SEQUENTIALLY</text>',
    '<text x="30" y="728" class="note">Derived from correctness.json, decode-runs.csv and serving-runs.csv; raw captures stay private.</text>',
    '</svg>']
out=base/'diagram/test-design.svg'
out.parent.mkdir(exist_ok=True)
out.write_text('\n'.join(parts)+'\n')
print(out)
