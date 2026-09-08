"""Numerical and interface checks beyond the timed power-of-two workloads."""
import json
import torch
from implementations import oracle
from triton_rmsnorm import fused_rmsnorm


def main():
    torch.manual_seed(123)
    checked = []
    for dtype in (torch.float32, torch.bfloat16):
        for rows, width in ((1, 1), (3, 33), (7, 1000), (32, 1024), (512, 4096), (2, 8192)):
            for pattern in ("random", "zero", "cancellation", "small"):
                x = torch.randn(rows, width, device="cuda", dtype=dtype)
                r = torch.randn_like(x)
                w = torch.randn(width, device="cuda", dtype=dtype)
                if pattern == "zero":
                    x.zero_(); r.zero_()
                elif pattern == "cancellation":
                    r = -x
                elif pattern == "small":
                    x.mul_(1e-4); r.mul_(1e-4)
                before = tuple(t.clone() for t in (x, r, w))
                y = fused_rmsnorm(x, r, w)
                expected = oracle(x, r, w, 1e-6)
                tol = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=.008, atol=.002)
                torch.testing.assert_close(y, expected, **tol)
                for a, b in zip(before, (x, r, w)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                checked.append(dict(dtype=str(dtype), rows=rows, width=width, pattern=pattern,
                                    max_abs_error=(y.float()-expected.float()).abs().max().item()))
    x = torch.randn(4, 8, device="cuda")
    for args in ((x.t(), x.t(), torch.ones(4, device="cuda")),
                 (x.requires_grad_(), x.detach(), torch.ones(8, device="cuda"))):
        try:
            fused_rmsnorm(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported layout/autograd input was accepted")
    print(json.dumps(dict(status="PASS", numerical_cases=checked, rejected_invalid_cases=2), indent=2))


if __name__ == "__main__":
    main()
