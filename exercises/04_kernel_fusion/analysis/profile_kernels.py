"""Record kernel counts separately from timing; do not export private trace paths."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from implementations import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with torch.inference_mode():
        for rows in [1, 4096]:
            data = (torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16),
                    torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16),
                    torch.randn(4096, device="cuda", dtype=torch.bfloat16))
            for method in ["eager", "compiled"]:
                fn = build(method, 1e-6)
                for _ in range(50):
                    fn(*data)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    output = fn(*data)
                    torch.cuda.synchronize()
                kernels = Counter(event.name for event in profile.events()
                                  if event.device_type == torch.autograd.DeviceType.CUDA
                                  and "memcpy" not in event.name.lower() and "memset" not in event.name.lower())
                results.append({"rows": rows, "width": 4096, "dtype": "bfloat16", "method": method,
                                "kernel_count": sum(kernels.values()), "kernels": dict(kernels)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"implementation_sha256": hashlib.sha256((ROOT / "src/implementations.py").read_bytes()).hexdigest(),
                                   "note": "One warmed ordinary call per case; profiling is separate from benchmark timing.",
                                   "profiles": results}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "kernels"} for row in results]))


if __name__ == "__main__":
    main()
