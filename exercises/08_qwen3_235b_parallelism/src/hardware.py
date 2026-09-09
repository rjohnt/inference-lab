"""Check two equal GPUs and measure a local peer-copy control before serving."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
Path(a.output).with_name('clock-anchor-start.json').write_text(json.dumps({'unix_s':time.time(),'perf_s':time.perf_counter()})+'\n')
assert torch.cuda.device_count()==2
props=[torch.cuda.get_device_properties(i) for i in range(2)]
assert props[0].name==props[1].name
peer=[torch.cuda.can_device_access_peer(0,1),torch.cuda.can_device_access_peer(1,0)]
assert all(peer), 'GPU peer access unavailable; inspect topology before benchmarking'
nbytes=256*2**20
src=torch.arange(nbytes//4,device='cuda:0',dtype=torch.float32)
dst=torch.empty_like(src,device='cuda:1')
dst.copy_(src)
assert torch.equal(src,dst.to('cuda:0'))
for _ in range(5): dst.copy_(src,non_blocking=True)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
times=[]
for _ in range(3):
    start=time.perf_counter()
    for _ in range(32): dst.copy_(src,non_blocking=True)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    times.append(time.perf_counter()-start)
d={'gpu_names':[p.name for p in props],'memory_bytes':[p.total_memory for p in props],
   'sm_counts':[p.multi_processor_count for p in props],'peer_access':peer,
   'torch':torch.__version__,'cuda':torch.version.cuda,
   'peer_copy_bytes':nbytes,'peer_copy_iterations':32,'peer_copy_wall_s':times,
   'peer_copy_gb_s':[nbytes*32/t/1e9 for t in times],
   'topology':subprocess.check_output(['nvidia-smi','topo','-m'],text=True)}
with open(a.output,'w') as f: json.dump(d,f,indent=2)
print(json.dumps({k:v for k,v in d.items() if k!='topology'},indent=2))
