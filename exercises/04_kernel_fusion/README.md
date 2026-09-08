# Residual + RMSNorm kernel fusion

Question: how much can a custom Triton kernel improve a transformer normalization
operation over eager and compiled PyTorch on an RTX 4070 SUPER?

Status: environment ready and verified on September 8, 2026. The fusion kernel
and performance study are planned; the vector-add check is infrastructure
validation only.

Verified: Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA runtime 13.0,
RTX 4070 SUPER (compute capability 8.9), host NVIDIA driver 591.74. Dependency
consistency (`pip check`), handwritten Triton FP32/FP16/BF16 addition, and
full-graph `torch.compile` all passed. See [verification results](results/environment-check.json)
and [resolved package versions](requirements.lock.txt).

## Setup and verification

Run from this directory on a Linux CUDA host with Python 3.12, a compatible NVIDIA
driver, and C/C++ compilers:

```bash
bash setup.sh
source env.sh
python src/smoke_test.py
```

Setup creates `.venv/` here, installs PyTorch 2.13.0 with CUDA 13.0 and the Triton
version required by that wheel, plus NumPy, pandas, Matplotlib, pytest, and Ninja.
`requirements.lock.txt` records the resolved versions. Compiler caches also stay
inside this directory and are ignored by Git.

Write and review source on the development machine; copy this exercise directory
to the GPU host. Compile and execute there. No PyTorch installation is required
on the editing machine. Host addresses and SSH credentials belong in local config.

The smoke test checks CUDA availability, a handwritten Triton addition kernel in
FP32/FP16/BF16 with a masked tail, and a full-graph `torch.compile` normalization
expression on two different inputs. Results go to `results/environment-check.json`.
The first-call duration includes compilation and must not be reported as latency
of a warmed kernel. `KERNEL_CHECK_OUTPUT` can redirect the check result.

## Planned experiment

1. Define the residual + RMSNorm reference and numerical tolerances.
2. Establish eager PyTorch and `torch.compile` baselines.
3. Write a correct fused Triton kernel; preserve that first implementation.
4. Tune block sizes and warp counts, recording every measured variant.
5. Compare latency distributions, speedups, and memory traffic across shapes.

Use fixed seeds, matching dtypes and semantics, correctness checks, separate
warmups, synchronized GPU timing, and randomized or alternating method order.
Include small decode-like and larger prefill-like workloads. Record regressions
as well as improvements. Profile representative cases to test the hypothesis:
fewer launches and intermediate memory transfers should help memory-bound cases.
Beating compiled PyTorch is an empirical question, not an assumption.

Keep future kernel source, shape configurations, compact measurements, charts,
and the final report in this directory.
