"""Draw logical topology from the tested configuration, not a kernel trace.
Run from anywhere: python3 analysis/topology.py. Rasterize with the diagram skill.
Dimensions denote unpacked logical weights in [output, input] order.
"""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[1]/'diagram'
OUT.mkdir(exist_ok=True)
C=['#22d3ee','#a78bfa']; BG='#0f172a'; MUTED='#a8b7cb'
def text(x,y,s,size=16,color='#e2e8f0',anchor='start',weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>'
def rect(x,y,w,h,color='#334155',fill='#172338',dash=False):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{color}" stroke-width="1.5"'+(' stroke-dasharray="7 5"' if dash else '')+'/>'
def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#fb923c" stroke-width="2" marker-end="url(#arrow)"/>'
def start(title,subtitle):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 850" role="img">',f'<title>{escape(title)}</title>', '<defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width="0.5"/></pattern><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#fb923c"/></marker></defs><style>text{font-family:ui-monospace,Menlo,Consolas,monospace}</style>',f'<rect width="1120" height="850" fill="{BG}"/><rect width="1120" height="850" fill="url(#grid)"/>',text(35,45,title,27,weight=700),text(35,78,subtitle,15,MUTED),rect(35,100,1050,40,'#fbbf24',BG),text(560,126,'Qwen3-235B AWQ  /  2 × A100 80GB  /  NV12 verified  /  one node',15,'#fbbf24','middle')]
for mode in ['tp','pp','ep']:
    title={'tp':'Tensor parallelism · split each matrix','pp':'Pipeline parallelism · split the layer stack','ep':'Expert parallelism · split the expert inventory'}[mode]
    subtitle={'tp':'Tested: TP=2 · PP=1 · EP off     |     Both ranks participate in every layer','pp':'Tested: TP=1 · PP=2 · EP off     |     A request crosses two sequential stages','ep':'Tested: TP=2 · PP=1 · EP on      |     Attention stays TP=2; experts become whole'}[mode]
    a=start(title,subtitle)
    # Connections are painted before boxes. These denote dependencies, not wire protocols.
    if mode=='pp':
        a += [arrow(450,410,670,410),text(560,366,'hidden state',14,'#fb923c','middle'),text(560,387,'+ residual',14,'#fb923c','middle'),text(560,446,'each [T,4096]',13,MUTED,'middle')]
    else:
        for y in [287,475]:a += [arrow(450,y,670,y),arrow(670,y+18,450,y+18)]
        a += [text(560,261,'combine partial',13,'#fb923c','middle'),text(560,279,'attention outputs',13,'#fb923c','middle'),text(560,449,'combine partial',13,'#fb923c','middle'),text(560,467,'expert outputs',13,'#fb923c','middle')]
    for rank,x in enumerate([35,670]):
        color=C[rank];a += [rect(x,160,415,495,color,BG),text(x+20,193,f'GPU {rank} · '+({'tp':f'TP rank {rank}','ep':f'TP / EP rank {rank}','pp':f'PP stage {rank}'}[mode]),21,color,weight=700)]
        layers=('layers 0–46' if rank==0 else 'layers 47–93') if mode=='pp' else 'layers 0–93 on both ranks'
        a += [text(x+20,219,layers,15,MUTED)]
        a += [rect(x+20,241,375,105,color),text(x+35,267,'ATTENTION',16,color,weight=700)]
        if mode=='pp':
            att=['All 64 Q heads + 4 KV heads','Full Q/K/V and output matrices']
        else:
            att=[f'32 Q heads + 2 KV heads per rank','Q/K/V output-axis shards','O projection input-axis shard']
        for j,s in enumerate(att):a.append(text(x+35,291+j*20,s,14))
        a += [rect(x+20,378,375,170,color),text(x+35,404,'MoE EXPERT WEIGHTS',16,color,weight=700)]
        if mode=='tp':
            a += [text(x+35,428,'All 128 experts; half of each',15)]
        elif mode=='ep':
            a += [text(x+35,428,f'64 whole experts: {rank*64}–{rank*64+63}',15)]
        else:a += [text(x+35,428,'All 128 whole experts per local layer',14)]
        # Representative expert glyphs: split tiles for TP; whole tiles otherwise.
        for i in range(6):
            tx=x+35+i*54
            a.append(rect(tx,441,43,31,'#475569',BG))
            if mode=='tp':
                a.append(f'<rect x="{tx+rank*21}" y="442" width="21" height="29" rx="3" fill="{color}" opacity="0.8"/>')
            else:a.append(f'<rect x="{tx+1}" y="442" width="41" height="29" rx="5" fill="{color}" opacity="0.7"/>')
        width=768 if mode=='tp' else 1536
        a += [text(x+35,496,f'gate/up: [{width},4096] each',14),text(x+35,519,f'down:    [4096,{width}]',14),text(x+35,537,'Logical shapes per local expert',12,MUTED)]
        a += [rect(x+20,580,375,55,'#34d399'),text(x+35,603,'BF16 KV CACHE · stays with attention',13,'#34d399'),text(x+35,625,'47 layers × 4 KV heads' if mode=='pp' else '94 layers × 2 KV heads',14)]
    if mode=='tp':
        notes=['Each expert has width 1536: rank 0 holds one 768-wide slice; rank 1 holds the other.', 'The router is replicated; the same top-8 expert selection is evaluated using both shards.']
    elif mode=='ep':
        notes=['Top-8 routing chooses among 128 experts; each rank contributes only its local experts.', 'Weights stay resident. Arrows show output-combination dependencies, not measured all-to-all.']
    else:
        notes=['Stage 0 precedes stage 1 for each forward pass; weights do not move at the stage boundary.', '47 / 47 is the vLLM default partition for 94 layers; this figure does not imply measured overlap.']
    a += [text(35,692,notes[0],15),text(35,718,notes[1],15),text(35,759,'Cyan = GPU 0 ownership    Violet = GPU 1 ownership    Orange = logical cross-rank dependency',13,MUTED),text(35,787,'Expert tiles are illustrative. AWQ packing/scales, embeddings, LM head and norms are omitted.',13,MUTED),text(35,813,'Logical layout derived from the tested config + vLLM 0.24 source; not a memory dump or kernel trace.',13,MUTED),'</svg>']
    (OUT/f'topology-{mode}.svg').write_text('\n'.join(a)+'\n')
