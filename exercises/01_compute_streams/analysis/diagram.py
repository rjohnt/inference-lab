from pathlib import Path
from html import escape
p=Path(__file__).resolve().parents[1]/'diagram/strategies.svg'
a=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 610"><defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern>
<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#94a3b8"/></marker>
<style>text{font-family:'SF Mono',monospace;fill:#e2e8f0}</style></defs>
<rect width="100%" height="100%" fill="#0f172a"/><rect width="100%" height="100%" fill="url(#grid)"/>''']
def text(x,y,t,size=12,anchor='start'):a.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(t)}</text>')
def rect(x,y,w,h,c):a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{c}" fill-opacity=".15" stroke="{c}"/>')
text(30,35,'Same four requests. Three ways to schedule the work.',20)
text(30,63,'Shared weight W • GPU-resident inputs • same outputs • conceptual, not measured timing',12)
for x,title,c in [(40,'Sequential','#94a3b8'),(350,'Multiple streams','#22d3ee'),(660,'Batched','#a78bfa')]:
 rect(x,100,260,380,c);text(x+130,130,title,17,'middle')
for i in range(4):
 rect(80,155+i*68,180,45,'#94a3b8');text(170,183+i*68,f'X{i} @ W',13,'middle')
 if i<3:a.append(f'<path d="M170 {200+i*68}V{219+i*68}" stroke="#94a3b8" marker-end="url(#arrow)"/>')
text(170,457,'One stream; ordered kernels',11,'middle')
for i in range(4):
 y=166+i*63;rect(370,y,220,44,'#22d3ee');text(480,y+27,f'Stream {i}: X{i} @ W',12,'middle')
text(480,457,'Independent branches; fork/join',11,'middle')
rect(680,195,220,175,'#a78bfa')
text(790,232,'[X0; X1; X2; X3]',15,'middle');text(790,274,'@ W',20,'middle');text(790,317,'One larger GEMM',13,'middle');text(790,347,'No packing in this test',10,'middle')
text(790,457,'Larger work unit; fewer launches',11,'middle')
text(40,522,'Streams permit concurrency; the GPU scheduler and available resources determine overlap.',12)
text(40,547,'Batching is natural for same-model requests, but a real server also pays queueing delay.',12)
text(40,572,'Measure ordinary calls and CUDA Graph replay; verify actual overlap in a separate trace.',12)
a.append('</svg>');p.write_text('\n'.join(a)+'\n')
