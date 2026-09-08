"""Verify CUDA, handwritten Triton, and full-graph torch.compile on GPUStation."""
import os
import json
import platform
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(output + offsets, tl.load(x + offsets, mask, other=0)
             + tl.load(y + offsets, mask, other=0), mask)


def expression(x, residual):
    z = x + residual
    return z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + 1e-6)


def main():
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.manual_seed(42)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Non-multiple size exercises masked loads/stores.
        x = torch.randn(100003, device="cuda", dtype=dtype)
        y = torch.randn_like(x)
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)
        torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    report["triton_add_fp32_fp16_bf16"] = "PASS"

    x = torch.randn(128, 2048, device="cuda")
    residual = torch.randn_like(x)
    compiled = torch.compile(expression, fullgraph=True)
    started = time.perf_counter()
    actual = compiled(x, residual)
    torch.cuda.synchronize()
    report["compile_and_first_run_seconds"] = time.perf_counter() - started
    torch.testing.assert_close(actual, expression(x, residual), rtol=1e-5, atol=1e-6)
    # Exercise cached execution with different inputs of the same shape.
    x2 = torch.randn_like(x)
    torch.testing.assert_close(compiled(x2, residual), expression(x2, residual),
                               rtol=1e-5, atol=1e-6)
    torch.cuda.synchronize()
    report["torch_compile_fullgraph"] = "PASS"
    output = Path(os.environ.get("KERNEL_CHECK_OUTPUT", str(Path(__file__).resolve().parents[1] / "results/environment-check.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
