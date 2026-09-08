# Residual + RMSNorm kernel fusion

Question: how much can a custom Triton kernel improve a transformer normalization
operation over eager and compiled PyTorch on an RTX 4070 SUPER?

Status: **eager versus `torch.compile` baseline complete**, September 8, 2026.
Two independent runs × 16 cases × 40 rounds per method and timing mode passed
correctness and result audits. The custom Triton fusion kernel is the next step.

[Full baseline report](results/baseline-v1/report.md) ·
[CSV timings](results/baseline-v1/timings.csv) ·
[SVG comparison](results/baseline-v1/speedups.svg) ·
[PNG comparison](results/baseline-v1/speedups.png)

![Eager versus compiled speedups](results/baseline-v1/speedups.svg)

Measured GPU execution speedups were **1.86–11.44×** in run 1. Ordinary Python-call
speedups ranged from **0.75× to 6.83×**: six FP32 cases regressed despite faster
GPU execution. The two runs' speedup ratios differed by at most 0.8% for GPU timing
and 3.8% for Python timing. These are warm-cache operation microbenchmarks, not
end-to-end model speedups.

For BF16 `[4096, 4096]`, GPU median time fell from 1,692.42 to 222.90 µs;
Python-call median fell from 1,710.96 to 250.63 µs. Separate profiling found
10 eager GPU kernels versus 1 compiled kernel in both checked BF16 shapes.
See [profile counts](results/baseline-v1/profile.json).

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
`requirements.lock.txt` pins the measured dependency versions; setup installs those pins without rewriting them. Compiler caches also stay
inside this directory and are ignored by Git.

Write and review source on the development machine; copy this exercise directory
to the GPU host. Compile and execute there. No PyTorch installation is required
on the editing machine. Host addresses and SSH credentials belong in local config.

The smoke test checks CUDA availability, a handwritten Triton addition kernel in
FP32/FP16/BF16 with a masked tail, and a full-graph `torch.compile` normalization
expression on two different inputs. Results go to `results/environment-check.json`.
The first-call duration includes compilation and must not be reported as latency
of a warmed kernel. `KERNEL_CHECK_OUTPUT` can redirect the check result.

## Sustained-sampling protocol (v2)

The initial 40-round runs above are exploratory baselines. The default is now
[long.json](configs/long.json): **at least 2,000 samples AND 30 measured seconds**
for every method/timing pair in every case. Sampling continues in balanced paired
cycles until every pair meets both floors (10,000-round fail-safe).

Each method is calibrated to a roughly 15 ms sample, then its batch size stays
fixed. Actual samples, calls/sample, total calls, and measured seconds are saved.
GPU duration is CUDA-event time; Python duration is synchronized wall time.
Calibration, compilation, correctness checks and warmup are excluded. There are
100 initial warmup calls plus 2 seconds of sustained warmup per method.

For 16 cases × 2 methods × 2 timing modes, the measured-duration floor is
32 minutes, plus setup and sampling overhead. The 2,000-sample floor is 50× the
initial sample count; call counts depend on measured operation speed. Confidence
intervals use paired 50-round block bootstrap to retain short-range timing
correlation. This is still a repeated-buffer microbenchmark; longer sampling
does not turn it into an end-to-end model or streaming-memory workload.

The duration controller passed a GPU smoke check and an independent sample audit.
The audit also rejects increased duration/sample requirements that the saved
smoke run did not satisfy. Long-run results will be published separately from v1.

## Repeat the benchmark

Run on the GPU host with other compute workloads idle:

```bash
source env.sh
python src/benchmark.py --config configs/long.json --out local/repeat1
python src/benchmark.py --config configs/long.json --out local/repeat2
python analysis/verify_run.py local/repeat1
python analysis/verify_run.py local/repeat2
python analysis/report.py local/repeat1 local/repeat2 --out local/repeat-report
python analysis/profile_kernels.py --out local/repeat-profile.json
```

Output directories must not already exist. A run writes `COMPLETE` only after
all cases pass. Run the profiler after timing is finished so it does not compete
with measurements. Original numeric samples stay under ignored `local/`; this
repository includes reviewed manifests, summaries, audits, CSV, and charts.
The report can also be regenerated from the published `results/baseline-v1/run1`
and `run2` directories (raw samples are needed only for the independent audit).

The [long-run configuration](configs/long.json) freezes seeds, epsilon, shapes, dtypes,
warmup, rounds, batching, and correctness tolerances. Every run records its exact
configuration, source hashes, environment and GPU-state snapshots. Compilation
and graph capture are outside timing. GPU timing uses identical CUDA Graph
capture for both methods; Python timing uses ordinary calls. All p95 values are
**p95 of per-call batch averages**, not per-request tail latency.

## Operation and extension contract

`src/implementations.py` defines the operation: add the residual in input dtype,
round that sum to input dtype, normalize and multiply by the weight in FP32, then
cast the normalized result back to input dtype. It returns only that result and
must not modify inputs. This weighted operation is more complete than the
unweighted setup smoke test. Forward inference, contiguous tensors, fixed shapes,
and finite random/zero inputs are the current scope.

To evaluate a later kernel, add a factory branch to `implementations.build` and
run it beside both controls:

```bash
# After implementing the `triton_fused` branch:
python src/benchmark.py --methods eager compiled triton_fused --out local/fusion1
```

Keep the operation contract and protocol fixed. Each implementation passes the
same FP64-reference and input-immutability checks before timing. Preserve earlier
run directories; publish a new results directory for each optimization. A changed
implementation has a new source hash. Repeat-run reports deliberately require
matching configurations and source hashes; compare optimizations using the
co-run eager/compiled controls and retain environment differences in the report.

The harness measures compilation/first-call time separately, but compiler caches
are persistent, so that number is not a controlled cold-compilation benchmark.
GPU clocks are unlocked and WSL shares the display GPU. Warm reused inputs may
fit in cache at small shapes; streaming-memory and end-to-end model tests remain
future work.

## Observer overhead

Elapsed-time checks, sample writes, and progress updates happen between timed
batches, after synchronization. They are excluded directly from per-call timing,
but can alter overall duty cycle or compete for host resources. SSH status reads
and GPU driver queries are also potential indirect disturbances, particularly
for the Python wall-clock metric. The current sustained run was checked
occasionally over SSH; GPU-state polling was stopped during the remaining
measurements after discussing observer overhead. No claim of zero observer
effect is made. A logging-disabled, unobserved repeat would be needed to measure
that effect directly.
