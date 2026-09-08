"""One warmed residual + RMSNorm invocation inside an NVTX profiling range."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-exercise", type=Path,
                        default=Path(__file__).resolve().parents[2] / "04_kernel_fusion")
    parser.add_argument("--method", choices=["eager", "compiled"], required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.rows < 1 or args.width < 1:
        parser.error("rows and width must be positive")
    source = args.kernel_exercise.resolve() / "src/implementations.py"
    if not source.is_file():
        parser.error("--kernel-exercise must contain src/implementations.py")
    sys.path.insert(0, str(source.parent))
    import torch
    from implementations import build, oracle

    assert torch.cuda.is_available(), "Use the CUDA-enabled kernel exercise Python environment"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tolerance = {"rtol": 0.008, "atol": 0.002} if dtype == torch.bfloat16 else {"rtol": 2e-5, "atol": 2e-6}
    with torch.inference_mode():
        data = (torch.randn(args.rows, args.width, device="cuda", dtype=dtype),
                torch.randn(args.rows, args.width, device="cuda", dtype=dtype),
                torch.randn(args.width, device="cuda", dtype=dtype))
        fn = build(args.method, 1e-6)
        expected = oracle(*data, 1e-6)
        for _ in range(100):
            output = fn(*data)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, **tolerance)
        torch.cuda.nvtx.range_push("rmsnorm")
        try:
            output = fn(*data)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.testing.assert_close(output, expected, **tolerance)
    print(json.dumps({"method": args.method, "rows": args.rows, "width": args.width,
                      "dtype": args.dtype, "seed": args.seed, "profiled_invocations": 1,
                      "correctness": "PASS", "torch": torch.__version__,
                      "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
