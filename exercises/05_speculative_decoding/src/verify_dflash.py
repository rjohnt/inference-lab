import hashlib,json,os
from pathlib import Path
root=Path('/path/to/data/gpustation-benchmark-models')
entries=json.loads((root/'dflash-tree.json').read_text())
entry=next(e for e in entries if e['path']=='Qwen3-8B-DFlash.Q8_0.gguf')
p=root/'dflash-q8.gguf'
h=hashlib.sha256()
with p.open('rb') as f:
    while True:
        b=f.read(8*1024*1024)
        if not b:break
        h.update(b)
    if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
assert p.stat().st_size==entry['size'],entry
assert h.hexdigest()==entry['lfs']['oid'],entry
link=Path.home()/'benchmarks/specdec/models/dflash-q8.gguf'
if not link.exists():link.symlink_to(p)
print(json.dumps({'size':p.stat().st_size,'sha256':h.hexdigest(),'path':str(p)}))
