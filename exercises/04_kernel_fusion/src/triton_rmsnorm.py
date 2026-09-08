"""Forward-only residual + weighted RMSNorm, one Triton program per row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, R, W, Y, N: tl.constexpr, EPS: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    r = tl.load(R + row * N + col, col < N, other=0).to(tl.float32)
    # Preserve PyTorch's intermediate BF16 rounding, not just the final cast.
    z = (x + r).to(Y.dtype.element_ty).to(tl.float32)
    variance = tl.sum(z * z, axis=0) / N
    inverse = tl.rsqrt(variance + EPS)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    tl.store(Y + row * N + col, (z * inverse) * w, col < N)


def fused_rmsnorm(x, residual, weight, eps=1e-6):
    if (x.ndim != 2 or residual.shape != x.shape or weight.shape != (x.shape[1],)
            or x.dtype not in (torch.float32, torch.bfloat16)
            or residual.dtype != x.dtype or weight.dtype != x.dtype
            or not x.is_cuda or residual.device != x.device or weight.device != x.device
            or not all(t.is_contiguous() for t in (x, residual, weight))
            or not 1 <= x.shape[1] <= 8192 or x.shape[0] < 1
            or any(t.requires_grad for t in (x, residual, weight))):
        raise ValueError("Expected contiguous CUDA FP32/BF16 inference tensors [M,N], [M,N], [N]; 1 <= N <= 8192")
    output = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    with torch.cuda.device(x.device):
        _rmsnorm[(x.shape[0],)](x, residual, weight, output, x.shape[1], eps,
                              block, num_warps=4 if block <= 4096 else 8,
                              enable_fp_fusion=False)
    return output
