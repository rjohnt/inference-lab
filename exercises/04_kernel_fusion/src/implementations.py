"""Stable operation contract and implementation registry for future optimizations."""
import torch


def residual_rmsnorm(x, residual, weight, eps=1e-6):
    # The residual addition rounds to the input dtype BEFORE FP32 normalization.
    z = (x + residual).float()
    scale = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps)
    return (z * scale * weight.float()).to(x.dtype)


def oracle(x, residual, weight, eps):
    """Independent FP64 reduction, with the same residual-rounding contract."""
    z = (x + residual).double()
    return (z / torch.sqrt(torch.sum(z * z, dim=-1, keepdim=True) / z.shape[-1] + eps)
            * weight.double()).to(x.dtype)


def build(name, eps):
    """Add future implementations here; each must accept (x, residual, weight)."""
    def reference(x, residual, weight):
        return residual_rmsnorm(x, residual, weight, eps)

    if name == "eager":
        return reference
    if name == "compiled":
        # Reset shape-specialization counters BETWEEN cases, never within timing.
        torch.compiler.reset()
        return torch.compile(reference, backend="inductor", fullgraph=True,
                             dynamic=False, options={"triton.cudagraphs": False})
    raise ValueError(f"Unknown implementation: {name}")
