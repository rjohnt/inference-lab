# Eager versus compiled residual + RMSNorm

## Question and prediction

Does Inductor compilation reduce warmed execution time for residual addition, RMS normalization, and learned scaling? Fewer launches and intermediate memory transfers should help, particularly on small workloads.

## Workload and environment

NVIDIA GeForce RTX 4070 SUPER; PyTorch 2.13.0+cu130; Triton 3.7.1; CUDA 13.0; driver 591.74; Python 3.12.3. CPU: AMD Ryzen 7 5800X 8-Core Processor. Linux/WSL, one PyTorch CPU thread. GPU clocks were not locked.

Forward inference only; contiguous row-major inputs; FP32 and BF16; residual addition rounds to input dtype before normalization; FP32 accumulation and scaling; output returns to input dtype. Only the normalized output is returned. This includes residual addition and weight scaling, not standalone RMSNorm.

## Method

2 independent process runs, 16 cases each, seed 20260908. Each method receives the same inputs, 50 warmup calls, and 40 measured rounds. Method order is randomized with balanced first/second positions; case order is deterministically shuffled. Inputs and allocations are reused across rounds (warm-cache microbenchmark).

GPU timing: CUDA events around 4 replays of a graph containing 32 calls, divided by total calls. Python timing: synchronized wall-clock batch of 32 ordinary calls, divided by batch size. Graph capture and compilation are excluded from both. Automatic compiler CUDA Graphs are disabled so graph treatment is matched between implementations.

Median and p95 below are distributions of **per-call batch averages**, not individual-request tail latency. Bootstrap confidence intervals in summaries resample paired rounds and describe within-run variability only. First-call durations are saved separately and may hit existing compiler caches; they are not cold compilation benchmarks.

Correctness: three random input seeds plus zeros for every method/shape, checked against an independent FP64 reduction with explicit dtype tolerances; input immutability, output dtype/shape, and captured-graph output are checked. These tests cover the benchmark inputs, not all possible numerical extremes.

## Results

![Speedups](speedups.svg)

### run1

| Case | Method | GPU eager µs p50 / p95 | GPU method µs p50 / p95 | GPU speedup | Python eager µs p50 / p95 | Python method µs p50 / p95 | Python speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| bfloat16-1x1024 | compiled | 10.04 / 10.05 | 0.92 / 0.92 | 10.91× | 78.08 / 80.42 | 73.71 / 76.09 | 1.06× |
| bfloat16-1x4096 | compiled | 11.61 / 11.62 | 1.44 / 1.44 | 8.06× | 76.42 / 79.79 | 71.86 / 74.86 | 1.06× |
| bfloat16-32x1024 | compiled | 13.09 / 13.12 | 1.14 / 1.14 | 11.44× | 77.14 / 79.58 | 71.12 / 73.07 | 1.08× |
| bfloat16-32x4096 | compiled | 16.22 / 16.30 | 1.54 / 1.54 | 10.56× | 77.66 / 79.28 | 71.30 / 72.75 | 1.09× |
| bfloat16-512x1024 | compiled | 20.45 / 20.47 | 2.43 / 2.45 | 8.41× | 79.92 / 84.33 | 72.53 / 81.89 | 1.10× |
| bfloat16-512x4096 | compiled | 52.66 / 53.48 | 5.82 / 5.96 | 9.04× | 80.59 / 84.00 | 72.60 / 77.51 | 1.11× |
| bfloat16-4096x1024 | compiled | 182.13 / 184.66 | 36.70 / 36.78 | 4.96× | 191.85 / 196.28 | 73.77 / 110.85 | 2.60× |
| bfloat16-4096x4096 | compiled | 1692.42 / 1712.44 | 222.90 / 225.24 | 7.59× | 1710.96 / 1735.49 | 250.63 / 255.88 | 6.83× |
| float32-1x1024 | compiled | 6.26 / 6.87 | 0.98 / 1.08 | 6.37× | 54.47 / 56.42 | 72.10 / 75.42 | 0.76× |
| float32-1x4096 | compiled | 6.67 / 6.69 | 1.60 / 1.60 | 4.17× | 54.14 / 55.64 | 71.47 / 73.03 | 0.76× |
| float32-32x1024 | compiled | 8.78 / 8.78 | 1.22 / 1.23 | 7.17× | 54.90 / 58.20 | 73.25 / 80.73 | 0.75× |
| float32-32x4096 | compiled | 12.00 / 12.01 | 1.94 / 1.94 | 6.20× | 55.23 / 56.99 | 71.78 / 75.23 | 0.77× |
| float32-512x1024 | compiled | 16.74 / 16.78 | 3.91 / 3.94 | 4.28× | 55.42 / 68.21 | 72.61 / 75.39 | 0.76× |
| float32-512x4096 | compiled | 41.84 / 41.97 | 12.73 / 12.82 | 3.29× | 58.27 / 60.93 | 73.04 / 76.27 | 0.80× |
| float32-4096x1024 | compiled | 185.89 / 188.69 | 99.77 / 99.84 | 1.86× | 197.96 / 199.98 | 117.35 / 119.77 | 1.69× |
| float32-4096x4096 | compiled | 1470.61 / 1486.54 | 452.46 / 453.85 | 3.25× | 1484.14 / 1504.03 | 472.29 / 479.04 | 3.14× |

### run2

| Case | Method | GPU eager µs p50 / p95 | GPU method µs p50 / p95 | GPU speedup | Python eager µs p50 / p95 | Python method µs p50 / p95 | Python speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| bfloat16-1x1024 | compiled | 10.03 / 10.04 | 0.91 / 0.92 | 11.00× | 78.70 / 85.91 | 73.35 / 78.85 | 1.07× |
| bfloat16-1x4096 | compiled | 10.54 / 10.56 | 1.31 / 1.32 | 8.04× | 75.65 / 83.29 | 70.23 / 73.04 | 1.08× |
| bfloat16-32x1024 | compiled | 12.02 / 12.02 | 1.05 / 1.05 | 11.47× | 76.73 / 78.27 | 70.10 / 71.74 | 1.09× |
| bfloat16-32x4096 | compiled | 16.24 / 16.30 | 1.53 / 1.54 | 10.58× | 78.54 / 79.40 | 69.50 / 70.70 | 1.13× |
| bfloat16-512x1024 | compiled | 20.42 / 20.43 | 2.42 / 2.44 | 8.42× | 79.54 / 82.29 | 71.79 / 76.23 | 1.11× |
| bfloat16-512x4096 | compiled | 52.48 / 53.44 | 5.80 / 5.96 | 9.04× | 81.29 / 85.99 | 73.36 / 75.75 | 1.11× |
| bfloat16-4096x1024 | compiled | 182.16 / 184.58 | 36.70 / 36.73 | 4.96× | 191.10 / 195.28 | 72.05 / 78.44 | 2.65× |
| bfloat16-4096x4096 | compiled | 1692.63 / 1712.38 | 222.89 / 224.96 | 7.59× | 1711.98 / 1733.13 | 250.57 / 253.99 | 6.83× |
| float32-1x1024 | compiled | 6.24 / 6.25 | 0.98 / 0.98 | 6.39× | 54.41 / 58.32 | 73.18 / 76.36 | 0.74× |
| float32-1x4096 | compiled | 6.67 / 6.68 | 1.60 / 1.60 | 4.17× | 53.05 / 57.62 | 69.74 / 75.54 | 0.76× |
| float32-32x1024 | compiled | 7.92 / 7.98 | 1.11 / 1.12 | 7.12× | 54.63 / 62.05 | 72.03 / 77.02 | 0.76× |
| float32-32x4096 | compiled | 12.00 / 12.02 | 1.93 / 1.94 | 6.22× | 55.10 / 57.59 | 72.43 / 76.32 | 0.76× |
| float32-512x1024 | compiled | 16.70 / 16.74 | 3.90 / 3.93 | 4.29× | 55.06 / 56.96 | 71.79 / 74.87 | 0.77× |
| float32-512x4096 | compiled | 41.75 / 41.97 | 12.70 / 12.82 | 3.29× | 58.17 / 61.85 | 72.59 / 75.11 | 0.80× |
| float32-4096x1024 | compiled | 185.79 / 186.05 | 99.76 / 99.91 | 1.86× | 197.63 / 201.65 | 118.46 / 125.19 | 1.67× |
| float32-4096x4096 | compiled | 1471.00 / 1487.90 | 452.32 / 452.88 | 3.25× | 1483.87 / 1502.65 | 471.56 / 476.47 | 3.15× |

## Profile evidence

Separate profiling of one warmed call per case (outside timed rounds):

| Shape / dtype | Method | GPU kernels |
|---|---|---:|
| 1 × 4096 / bfloat16 | eager | 10 |
| 1 × 4096 / bfloat16 | compiled | 1 |
| 4096 × 4096 / bfloat16 | eager | 10 |
| 4096 × 4096 / bfloat16 | compiled | 1 |

[Kernel names and counts](profile.json). This supports the launch-fusion explanation; no hardware-counter measurement of DRAM traffic was performed.

## Interpretation and limitations

These measurements establish a local baseline for later custom kernels. The graph result isolates device execution more closely; the Python result also reflects dispatch, allocation and synchronization overhead. Neither is end-to-end model throughput. Reused inputs may fit in GPU cache for small shapes; do not extrapolate these ratios to streaming-memory workloads.

The second process run checks immediate repeatability on the same machine, not cross-day or cross-machine reproducibility. Windows display activity, thermal state, power management, and unlocked clocks can affect results. Per-case GPU snapshots are retained in summaries. Do not average the two timing modes together.

## Reproduction and later optimizations

From the exercise directory on the GPU host:

```bash
source env.sh
python src/benchmark.py --config configs/baseline.json --out local/new-run1
python src/benchmark.py --config configs/baseline.json --out local/new-run2
python analysis/report.py local/new-run1 local/new-run2 --out local/new-report
```

Output directories must be new. `COMPLETE` is written only after all cases pass. Raw numeric samples stay in ignored `local/`; reviewed manifests, summaries, CSV and charts can be published. Source/configuration SHA256 values identify the measured experiment. Add later kernels through `implementations.build`, keep the operation and protocol fixed, and include both eager and compiled baselines in each new run.
