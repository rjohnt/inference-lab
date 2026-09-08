"""Generate the explanatory SVG; conceptual timing is deliberately not measured."""
from pathlib import Path
from html import escape
out=Path(__file__).resolve().parents[1]/'diagram/theory.svg'
a=['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 710"><defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern>
<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#94a3b8"/></marker>
<style>text{font-family:'SF Mono','Cascadia Code',monospace;fill:#e2e8f0} .sub{fill:#94a3b8}</style></defs>
<rect width="100%" height="100%" fill="#0f172a"/><rect width="100%" height="100%" fill="url(#grid)"/>''']
def text(x,y,s,size=12,anchor='start',color=None):
 a.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'+(f' style="fill:{color}"' if color else '')+'>'+escape(s)+'</text>')
def box(x,y,w,h,title,sub,c):
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#0f172a"/>')
 a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{c}" fill-opacity=".12" stroke="{c}"/>')
 text(x+w/2,y+27,title,13,'middle');text(x+w/2,y+47,sub,10,'middle','#94a3b8')
text(30,35,'Keeping the GPU fed',22);text(30,59,'Conceptual model — bar lengths are illustrative, not benchmark measurements',12,color='#94a3b8')
for x1,x2,label in [(260,390,'H2D upload'),(610,740,'D2H download')]:
 a.append(f'<path d="M{x1} 143H{x2}" stroke="#94a3b8" marker-end="url(#arrow)"/>');text((x1+x2)/2,124,label,11,'middle')
box(40,110,220,70,'CPU input buffers','Pinned: reusable DMA source','#22d3ee')
box(390,110,220,70,'GPU memory + compute','Weights remain on the GPU','#34d399')
box(740,110,220,70,'CPU output buffers','Read after transfer completion','#a78bfa')
text(40,219,'Serial: finish one batch before starting the next',15)
colors=['#22d3ee','#34d399','#a78bfa'];labels=['Upload','Compute','Download']
for b in range(2):
 for k in range(3):
  x=180+b*330+k*100
  a.append(f'<rect x="{x}" y="240" width="96" height="36" rx="4" fill="{colors[k]}" fill-opacity=".25" stroke="{colors[k]}"/>');text(x+48,262,f'{labels[k]} {b}',11,'middle')
text(40,319,'Pipelined: different batches use different stages concurrently',15)
for k in range(3):
 text(45,363+k*50,labels[k],12)
 for b in range(4):
  x=180+(b+k)*100;y=340+k*50
  a.append(f'<rect x="{x}" y="{y}" width="96" height="34" rx="4" fill="{colors[k]}" fill-opacity=".25" stroke="{colors[k]}"/>');text(x+48,y+22,f'Batch {b}',11,'middle')
text(180,501,'Time →   fill pipeline  ·  steady state  ·  drain pipeline',12,color='#94a3b8')
text(40,553,'Serial batch time ≈ upload + compute + download',15)
text(40,582,'Ideal pipelined interval ≈ max(upload, compute, download)',15)
text(40,621,'Actual overlap is limited by copy engines, memory bandwidth and CPU submission.',11,color='#94a3b8')
text(40,644,'CUDA events enforce dependencies. A slot is reused only after its download finishes.',11,color='#94a3b8')
text(40,673,'GPU-resident control removes transfers; pinned memory alone does not create overlap.',11,color='#94a3b8')
a.append('</svg>');out.write_text('\n'.join(a)+'\n')
