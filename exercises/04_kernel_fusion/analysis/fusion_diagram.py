"""Conceptual dataflow for the exact residual-rounding contract."""
from html import escape
from pathlib import Path
out = Path(__file__).resolve().parents[1] / "diagram/fusion.svg"
out.parent.mkdir(parents=True, exist_ok=True)
a = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 590"><defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern>
<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#94a3b8"/></marker>
<style>text{font-family:'SF Mono','Cascadia Code',monospace;fill:#e2e8f0}</style></defs>
<rect width="100%" height="100%" fill="#0f172a"/><rect width="100%" height="100%" fill="url(#grid)"/>''']
def text(x,y,s,size=14,color="#e2e8f0",anchor="start"):
    a.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" text-anchor="{anchor}">{escape(s)}</text>')
def box(x,y,w,title,sub,color):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="70" rx="8" fill="#0f172a"/>')
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="70" rx="8" fill="{color}" fill-opacity=".12" stroke="{color}"/>')
    text(x+w/2,y+28,title,15,anchor="middle");text(x+w/2,y+49,sub,11,"#94a3b8","middle")
text(30,36,"Residual + weighted RMSNorm: one operation, three implementations",21)
text(30,62,"Conceptual dataflow — box widths do not represent latency or DRAM traffic",13,"#94a3b8")
for x1,x2 in ((265,305),(520,560),(795,835)):
    a.append(f'<path d="M{x1} 150H{x2}" stroke="#94a3b8" marker-end="url(#arrow)"/>')
box(35,115,230,"Residual addition","z = round_dtype(x + r)","#22d3ee")
box(305,115,215,"FP32 row reduction","mean(z × z)","#34d399")
box(560,115,235,"Normalize and weight","z × rsqrt(mean + eps) × w","#34d399")
box(835,115,230,"Output","cast to input dtype","#a78bfa")
text(35,230,"Eager PyTorch",18,"#22d3ee")
text(35,258,"Separate operations allocate intermediates and launch multiple GPU kernels.",14)
text(35,285,"BF16 residual rounding is part of the contract, even when everything is fused.",13,"#fbbf24")
box(35,325,490,"torch.compile / Inductor","Generates fused code for the same Python expression","#a78bfa")
box(570,325,495,"Custom Triton","One row per program; masked loads; FP32 reduction","#34d399")
text(35,440,"Fusion alone does not establish an advantage over torch.compile.",17)
text(35,474,"Measure GPU execution and ordinary Python calls separately.",14,"#94a3b8")
text(35,504,"Identical inputs, FP64 reference, no input mutation; compilation outside timing.",13,"#94a3b8")
text(35,544,"Warm reused buffers test this microbenchmark, not full-model serving throughput.",13,"#94a3b8")
a.append('</svg>');out.write_text('\n'.join(a)+'\n')
