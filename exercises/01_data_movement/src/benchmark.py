"""Host/device movement and bounded three-stream pipeline; run on a CUDA host."""
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / '04_kernel_fusion' / 'src'))
from implementations import build, oracle


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def timed(fn, count):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn(count)
    torch.cuda.synchronize()
    return time.perf_counter() - t


def measure(fn, seconds, cfg, fixed_count=None):
    # Calibration and warmup excluded; count fixed for the entire case.
    timed(fn, 2)
    probe = timed(fn, 4)
    count = fixed_count or max(1, min(4096, math.ceil(4 * cfg['target_sample_seconds'] / probe)))
    timed(fn, count)
    samples = []
    elapsed = 0.0
    while len(samples) < cfg['min_samples'] or elapsed < seconds:
        if len(samples) >= cfg['max_samples']:
            raise RuntimeError('sampling cap reached before both floors')
        duration = timed(fn, count)
        samples.append(duration)
        elapsed += duration
    return dict(count=count, samples=len(samples), measured_seconds=elapsed,
                median_seconds=float(np.median(samples)) / count,
                p95_seconds=float(np.percentile(samples, 95)) / count,
                total_operations=count * len(samples)), samples


class Pipeline:
    def __init__(self, workload, variant, cfg):
        self.variant, self.workload = variant, workload
        self.slots = cfg['slots']
        pinned = variant != 'pageable_serial'
        rows, width = cfg[workload + '_shape']
        self.shape = (rows, width)
        self.h = [torch.randn((2, rows, width) if workload == 'rmsnorm' else (rows, width),
                              dtype=torch.bfloat16).pin_memory() if pinned else
                  torch.randn((2, rows, width) if workload == 'rmsnorm' else (rows, width),
                              dtype=torch.bfloat16) for _ in range(self.slots)]
        self.d = [x.to('cuda') for x in self.h]
        self.out = [torch.empty((rows, width), dtype=torch.bfloat16, pin_memory=pinned)
                    for _ in range(self.slots)]
        self.weight = (torch.randn(width, dtype=torch.bfloat16, device='cuda') if workload == 'rmsnorm'
                       else torch.randn(width, width, dtype=torch.bfloat16, device='cuda') / math.sqrt(width))
        self.fn = build('compiled', 1e-6) if workload == 'rmsnorm' else None
        self.y = [self.compute(x) for x in self.d]
        self.expected = [(oracle(x[0], x[1], self.weight, 1e-6) if workload == 'rmsnorm'
                          else (x.float() @ self.weight.float()).bfloat16()).cpu() for x in self.d]
        self.streams = [torch.cuda.Stream() for _ in range(3)]
        self.ready = [torch.cuda.Event() for _ in range(self.slots)]
        self.computed = [torch.cuda.Event() for _ in range(self.slots)]
        self.done = [torch.cuda.Event() for _ in range(self.slots)]
        torch.cuda.synchronize()
        self.upload_bytes = self.h[0].numel() * self.h[0].element_size()
        self.download_bytes = self.out[0].numel() * self.out[0].element_size()

    def compute(self, x):
        return self.fn(x[0], x[1], self.weight) if self.workload == 'rmsnorm' else x @ self.weight

    def run(self, count, trace=False):
        records = []
        origin = torch.cuda.Event(enable_timing=True) if trace else None
        if trace:
            origin.record()
            for stream in self.streams:
                stream.wait_event(origin)
        def stage(name, batch, action):
            if not trace:
                action()
                return
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            with torch.cuda.nvtx.range(f'{name}/batch{batch}'):
                action()
            b.record()
            records.append((name, batch, a, b))
        def compute_slot(i):
            self.y[i] = self.compute(self.d[i])
        for batch in range(count):
            i = batch % self.slots
            if self.variant == 'overlapped':
                # Host waits before reusing a slot, retaining tensor storage until D2H completes.
                if batch >= self.slots:
                    self.done[i].synchronize()
                with torch.cuda.stream(self.streams[0]):
                    stage('H2D', batch, lambda: self.d[i].copy_(self.h[i], non_blocking=True))
                    self.ready[i].record()
                with torch.cuda.stream(self.streams[1]):
                    self.streams[1].wait_event(self.ready[i])
                    stage('compute', batch, lambda: compute_slot(i))
                    self.computed[i].record()
                with torch.cuda.stream(self.streams[2]):
                    self.streams[2].wait_event(self.computed[i])
                    stage('D2H', batch, lambda: self.out[i].copy_(self.y[i], non_blocking=True))
                    self.done[i].record()
            elif self.variant == 'resident':
                stage('compute', batch, lambda: compute_slot(i))
            else:
                stage('H2D', batch, lambda: self.d[i].copy_(self.h[i], non_blocking=self.variant == 'pinned_serial'))
                stage('compute', batch, lambda: compute_slot(i))
                stage('D2H', batch, lambda: self.out[i].copy_(self.y[i], non_blocking=self.variant == 'pinned_serial'))
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        if trace:
            return [dict(stage=n, batch=j, start_ms=origin.elapsed_time(a), end_ms=origin.elapsed_time(b))
                    for n, j, a, b in records]

    def verify(self):
        self.run(self.slots * 3 + 1)
        for i, expected in enumerate(self.expected):
            actual = self.y[i].cpu() if self.variant == 'resident' else self.out[i]
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)


def transfer(size, direction, pinned, chunks=1):
    source = torch.randint(0, 251, (size,), dtype=torch.uint8)
    host = torch.empty(size, dtype=torch.uint8, pin_memory=pinned)
    host.copy_(source)
    device = host.to('cuda')
    if direction == 'H2D':
        device.zero_()
    else:
        host.zero_()
    pairs = list(zip(device.chunk(chunks), host.chunk(chunks)))
    def run(count):
        for _ in range(count):
            for d, h in pairs:
                (d.copy_(h, non_blocking=True) if direction == 'H2D'
                 else h.copy_(d, non_blocking=True))
    timed(run, 1)
    torch.testing.assert_close(device.cpu() if direction == 'H2D' else host, source, rtol=0, atol=0)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=ROOT/'configs/standard.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--profile', choices=['pageable_serial', 'pinned_serial', 'overlapped', 'resident'])
    ap.add_argument('--workload', choices=['rmsnorm', 'matmul'], default='matmul')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg.update(repeats=1, transfer_bytes=[4096,1048576], transfer_seconds=.05,
                   pipeline_seconds=.05, min_samples=3, pipeline_batches=6)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.profile:
        p = Pipeline(args.workload, args.profile, cfg)
        p.verify()
        p.run(24)
        torch.cuda.cudart().cudaProfilerStart()
        events = p.run(24, trace=True)
        torch.cuda.cudart().cudaProfilerStop()
        save(args.out/'timeline.json', dict(workload=args.workload, variant=args.profile, events=events,
             timing='CUDA events in a separate instrumented run; stage spans include scheduling gaps'))
        return
    manifest = dict(config=cfg, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operation_sha256=hashlib.sha256((ROOT.parent/'04_kernel_fusion/src/implementations.py').read_bytes()).hexdigest(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    save(args.out/'manifest.json', manifest)
    results = []
    with (args.out/'samples.jsonl').open('w') as raw:
        for rep in range(cfg['repeats']):
            cases = [('transfer', size, direction, pinned, 1) for size in cfg['transfer_bytes']
                     for direction in ['H2D','D2H'] for pinned in [False,True]]
            cases += [('packing', cfg['packing_bytes'], 'H2D', True, n) for n in cfg['packing_chunks']]
            cases += [('pipeline', w, v) for w in ['rmsnorm','matmul']
                      for v in ['pageable_serial','pinned_serial','overlapped','resident']]
            random.Random(cfg['seed']+rep).shuffle(cases)
            for index, case in enumerate(cases):
                kind = case[0]
                torch.manual_seed(cfg['seed'])
                if kind == 'pipeline':
                    _, workload, variant = case
                    p = Pipeline(workload, variant, cfg)
                    p.verify()
                    row, samples = measure(p.run, cfg['pipeline_seconds'], cfg, cfg['pipeline_batches'])
                    p.verify()
                    meta = dict(workload=workload, variant=variant, upload_bytes=p.upload_bytes,
                                download_bytes=p.download_bytes)
                    del p
                else:
                    _, size, direction, pinned, chunks = case
                    fn = transfer(size, direction, pinned, chunks)
                    row, samples = measure(fn, cfg['transfer_seconds'], cfg)
                    meta = dict(bytes=size, direction=direction, pinned=pinned, chunks=chunks)
                    del fn
                item = dict(repeat=rep+1, kind=kind, **meta, **row, correctness='PASS')
                results.append(item)
                raw.write(json.dumps(dict(case=len(results)-1, seconds=samples))+'\n')
                raw.flush()
                save(args.out/'summary.json',results)
                save(args.out/'progress.json',dict(repeat=rep+1, case=index+1,total=len(cases)))
    manifest['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(args.out/'manifest.json',manifest)
    (args.out/'COMPLETE').write_text('All cases completed; numerical checks passed.\n')

if __name__ == '__main__':
    with torch.inference_mode():
        main()
