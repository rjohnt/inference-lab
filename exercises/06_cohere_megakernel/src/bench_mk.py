"""Run unmodified upstream decode and save synchronized native-loop wall time."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--context', type=int, required=True)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--tokens', type=int, default=512)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.root / 'megakernel/src'))
import torch
import runner

torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
args = runner._parse_args([
    '--mk', '--fast', '--real-weight', '--checkpoint', str(a.root / 'checkpoint'),
    '--lib-path', str(a.root / 'megakernel/build/libmk_release.so'),
    '--batch-size', str(a.batch), '--max-new-tokens', str(a.tokens),
    '--fake-prompt-len', str(a.context), '--num-sms', '132',
    '--frac-vram-utilization', '0.90', '--cpp-decode-runtime', '--temperature', '0',
])
torch.cuda.reset_peak_memory_stats()
report = runner._run_native_benchmark(args, runner.SamplingParams(max_new_tokens=a.tokens, temperature=0.0))
result = report.result
steps = [float(x) for x in result['decode_step_ms']]
assert len(steps) == a.tokens - 1, (len(steps), a.tokens)
assert all(len(row) == a.tokens for row in result['token_ids_by_batch'])
data = {
    'engine': 'megakernel', 'context': a.context, 'batch': a.batch, 'seed': a.seed,
    'output_tokens_per_request': a.tokens, 'decode_steps': len(steps),
    'native_decode_wall_ms': sum(steps), 'tpot_ms': statistics.mean(steps),
    'decode_tok_s': a.batch * len(steps) * 1000 / sum(steps),
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    'timing_scope': 'C++ decode loop, excludes loading, schedule/JIT, synthetic KV fill and bootstrap token',
    'timing_method': 'total synchronized loop wall / executed steps; upstream step values repeat the loop average',
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k != 'step_ms'}), flush=True)
