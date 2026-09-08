"""Warm and capture the benchmark pipeline without extra per-stage timing events."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from benchmark import Pipeline, ROOT, save

ap = argparse.ArgumentParser()
ap.add_argument('--workload', choices=['rmsnorm', 'matmul'], required=True)
ap.add_argument('--variant', choices=['pinned_serial', 'overlapped'], required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
cfg = json.loads((ROOT/'configs/standard.json').read_text())
args.out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
torch.manual_seed(cfg['seed'])
torch.backends.cuda.matmul.allow_tf32 = False
with torch.inference_mode():
    pipeline = Pipeline(args.workload, args.variant, cfg)
    pipeline.verify()
    pipeline.run(24)
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range(f'{args.workload}/{args.variant}'):
        pipeline.run(24)
    torch.cuda.cudart().cudaProfilerStop()
save(args.out/'capture.json', dict(workload=args.workload, variant=args.variant, batches=24,
     benchmark_sha256=hashlib.sha256((ROOT/'src/benchmark.py').read_bytes()).hexdigest(),
     profile_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     config=cfg, correctness='PASS', notes='Same pipeline path as benchmark; no additional per-stage CUDA timing events.'))
