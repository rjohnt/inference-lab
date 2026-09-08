# Residual + RMSNorm implementation comparison

## Question and prediction

How do eager PyTorch, Inductor compilation and any registered custom kernels compare on residual addition, RMS normalization and learned scaling? Fewer launches and intermediate allocations may help; a handwritten kernel is not guaranteed to beat compiler-generated code.

## Workload and environment

NVIDIA GeForce RTX 4070 SUPER; PyTorch 2.13.0+cu130; Triton 3.7.1; CUDA 13.0; driver 591.74; Python 3.12.3. CPU: AMD Ryzen 7 5800X 8-Core Processor. Linux/WSL, one PyTorch CPU thread. GPU clocks were not locked.

Forward inference only; contiguous row-major inputs; FP32 and BF16; residual addition rounds to input dtype before normalization; FP32 accumulation and scaling; output returns to input dtype. Only the normalized output is returned. This includes residual addition and weight scaling, not standalone RMSNorm.

## Method

1 independent process runs, 16 cases each, seed 20260908. Each method receives the same inputs, 100 warmup calls, and at least 2000 measured rounds. Method order is randomized with balanced execution positions; case order is deterministically shuffled. Inputs and allocations are reused across rounds (warm-cache microbenchmark).

At least 2,000 rounds AND 30 measured seconds per method/timing pair per case. Batches are calibrated once per method toward 15 ms, then held fixed. Additional warmup: 2 seconds per method. Final round counts, batch calls and measured durations are recorded in summaries. Confidence intervals use paired circular block bootstrap with 50-round blocks to preserve short-range timing correlation.

GPU timing uses CUDA events around repeated replay of a graph containing 32 calls, divided by all replayed calls. Python timing uses a synchronized batch of ordinary calls divided by batch size. Graph capture, calibration and compilation are excluded. Automatic compiler CUDA Graphs are disabled so graph treatment is matched between implementations.

Median and p95 below are distributions of **per-call batch averages**, not individual-request tail latency. Bootstrap confidence intervals in summaries resample paired rounds and describe within-run variability only. First-call durations are saved separately and may hit existing compiler caches; they are not cold compilation benchmarks.

Correctness: three random input seeds plus zeros for every method/shape, checked against an independent FP64 reduction with explicit dtype tolerances; input immutability, output dtype/shape, and captured-graph output are checked. These tests cover the benchmark inputs, not all possible numerical extremes.

## Results

![Speedups](speedups.png)

[SVG](speedups.svg)

### run1

| Case | Method | GPU eager µs p50 / p95 | GPU method µs p50 / p95 | GPU speedup | Python eager µs p50 / p95 | Python method µs p50 / p95 | Python speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| bfloat16-1x1024 | compiled | 10.21 / 10.28 | 0.92 / 0.92 | 11.14× | 72.96 / 75.92 | 68.54 / 71.77 | 1.06× |
| bfloat16-1x1024 | triton_fused | 10.21 / 10.28 | 1.02 / 1.02 | 10.04× | 72.96 / 75.92 | 27.51 / 28.38 | 2.65× |
| bfloat16-1x4096 | compiled | 10.63 / 10.68 | 1.31 / 1.32 | 8.08× | 72.81 / 74.75 | 67.74 / 69.88 | 1.07× |
| bfloat16-1x4096 | triton_fused | 10.63 / 10.68 | 1.42 / 1.42 | 7.50× | 72.81 / 74.75 | 27.33 / 27.98 | 2.66× |
| bfloat16-32x1024 | compiled | 12.02 / 12.07 | 1.04 / 1.05 | 11.53× | 73.78 / 76.40 | 68.50 / 71.30 | 1.08× |
| bfloat16-32x1024 | triton_fused | 12.02 / 12.07 | 1.14 / 1.15 | 10.52× | 73.78 / 76.40 | 27.37 / 28.17 | 2.70× |
| bfloat16-32x4096 | compiled | 16.26 / 16.32 | 1.53 / 1.53 | 10.65× | 75.90 / 77.66 | 68.85 / 71.40 | 1.10× |
| bfloat16-32x4096 | triton_fused | 16.26 / 16.32 | 1.66 / 1.66 | 9.80× | 75.90 / 77.66 | 28.09 / 28.69 | 2.70× |
| bfloat16-512x1024 | compiled | 20.59 / 20.64 | 2.44 / 2.44 | 8.46× | 75.10 / 77.60 | 67.82 / 70.57 | 1.11× |
| bfloat16-512x1024 | triton_fused | 20.59 / 20.64 | 2.16 / 2.16 | 9.54× | 75.10 / 77.60 | 27.79 / 28.55 | 2.70× |
| bfloat16-512x4096 | compiled | 53.10 / 55.77 | 5.86 / 6.22 | 9.06× | 75.01 / 77.19 | 68.64 / 71.03 | 1.09× |
| bfloat16-512x4096 | triton_fused | 53.10 / 55.77 | 7.02 / 7.41 | 7.56× | 75.01 / 77.19 | 27.88 / 28.53 | 2.69× |
| bfloat16-4096x1024 | compiled | 182.26 / 183.72 | 36.37 / 36.44 | 5.01× | 187.36 / 190.16 | 69.16 / 71.50 | 2.71× |
| bfloat16-4096x1024 | triton_fused | 182.26 / 183.72 | 9.00 / 9.20 | 20.26× | 187.36 / 190.16 | 28.49 / 29.21 | 6.58× |
| bfloat16-4096x4096 | compiled | 1692.35 / 1712.29 | 223.38 / 225.04 | 7.58× | 1726.34 / 1749.02 | 243.12 / 246.65 | 7.10× |
| bfloat16-4096x4096 | triton_fused | 1692.35 / 1712.29 | 208.21 / 213.40 | 8.13× | 1726.34 / 1749.02 | 217.16 / 221.20 | 7.95× |
| float32-1x1024 | compiled | 6.25 / 6.29 | 0.98 / 0.98 | 6.40× | 49.93 / 51.70 | 68.40 / 72.02 | 0.73× |
| float32-1x1024 | triton_fused | 6.25 / 6.29 | 1.07 / 1.08 | 5.85× | 49.93 / 51.70 | 27.44 / 28.46 | 1.82× |
| float32-1x4096 | compiled | 6.66 / 6.71 | 1.59 / 1.59 | 4.19× | 49.71 / 51.42 | 69.25 / 71.95 | 0.72× |
| float32-1x4096 | triton_fused | 6.66 / 6.71 | 1.59 / 1.60 | 4.18× | 49.71 / 51.42 | 27.31 / 27.88 | 1.82× |
| float32-32x1024 | compiled | 7.93 / 7.96 | 1.11 / 1.12 | 7.14× | 50.65 / 52.33 | 68.00 / 70.54 | 0.74× |
| float32-32x1024 | triton_fused | 7.93 / 7.96 | 1.22 / 1.23 | 6.50× | 50.65 / 52.33 | 27.32 / 27.92 | 1.85× |
| float32-32x4096 | compiled | 11.98 / 12.04 | 1.93 / 1.94 | 6.20× | 51.65 / 53.35 | 69.21 / 71.96 | 0.75× |
| float32-32x4096 | triton_fused | 11.98 / 12.04 | 1.97 / 1.98 | 6.08× | 51.65 / 53.35 | 27.91 / 28.63 | 1.85× |
| float32-512x1024 | compiled | 15.27 / 15.30 | 3.56 / 3.58 | 4.28× | 50.56 / 52.13 | 68.23 / 71.22 | 0.74× |
| float32-512x1024 | triton_fused | 15.27 / 15.30 | 3.66 / 3.67 | 4.17× | 50.56 / 52.13 | 27.14 / 27.92 | 1.86× |
| float32-512x4096 | compiled | 42.19 / 43.30 | 12.88 / 13.27 | 3.27× | 54.31 / 55.83 | 68.62 / 71.55 | 0.79× |
| float32-512x4096 | triton_fused | 42.19 / 43.30 | 12.23 / 12.68 | 3.45× | 54.31 / 55.83 | 28.18 / 29.06 | 1.93× |
| float32-4096x1024 | compiled | 185.75 / 186.07 | 99.72 / 99.78 | 1.86× | 193.94 / 196.74 | 113.30 / 116.12 | 1.71× |
| float32-4096x1024 | triton_fused | 185.75 / 186.07 | 111.51 / 112.11 | 1.67× | 193.94 / 196.74 | 119.14 / 120.44 | 1.63× |
| float32-4096x4096 | compiled | 1470.75 / 1488.51 | 452.91 / 455.74 | 3.25× | 1497.63 / 1517.26 | 470.90 / 476.41 | 3.18× |
| float32-4096x4096 | triton_fused | 1470.75 / 1488.51 | 433.70 / 444.54 | 3.39× | 1497.63 / 1517.26 | 453.52 / 462.54 | 3.30× |

## Run duration

| Run | Timestamp-to-timestamp elapsed minutes | Sum of measured batch minutes |
|---|---:|---:|
| run1 | 58.28 | 55.43 |

Elapsed timestamps include recorded setup, calibration, checks, reporting and gaps; measured totals sum the distinct timed batches. Process imports before the start timestamp are not included.

## Actual sampling totals

| Run | Case | Metric | Method | Samples | Calls/sample | Total calls | Measured seconds |
|---|---|---|---|---:|---:|---:|---:|
| run1 | bfloat16-1x1024 | graph_gpu | eager | 2196 | 1376 | 3,021,696 | 30.88 |
| run1 | bfloat16-1x1024 | graph_gpu | compiled | 2196 | 14912 | 32,746,752 | 30.04 |
| run1 | bfloat16-1x1024 | graph_gpu | triton_fused | 2196 | 13440 | 29,514,240 | 30.04 |
| run1 | bfloat16-1x1024 | wall | eager | 2196 | 205 | 450,180 | 33.08 |
| run1 | bfloat16-1x1024 | wall | compiled | 2196 | 218 | 478,728 | 33.04 |
| run1 | bfloat16-1x1024 | wall | triton_fused | 2196 | 494 | 1,084,824 | 30.00 |
| run1 | bfloat16-1x4096 | graph_gpu | eager | 2187 | 1312 | 2,869,344 | 30.53 |
| run1 | bfloat16-1x4096 | graph_gpu | compiled | 2187 | 10432 | 22,814,784 | 30.02 |
| run1 | bfloat16-1x4096 | graph_gpu | triton_fused | 2187 | 9696 | 21,205,152 | 30.07 |
| run1 | bfloat16-1x4096 | wall | eager | 2187 | 205 | 448,335 | 32.78 |
| run1 | bfloat16-1x4096 | wall | compiled | 2187 | 217 | 474,579 | 32.32 |
| run1 | bfloat16-1x4096 | wall | triton_fused | 2187 | 511 | 1,117,557 | 30.69 |
| run1 | bfloat16-32x1024 | graph_gpu | eager | 2199 | 1152 | 2,533,248 | 30.44 |
| run1 | bfloat16-32x1024 | graph_gpu | compiled | 2199 | 13120 | 28,850,880 | 30.06 |
| run1 | bfloat16-32x1024 | graph_gpu | triton_fused | 2199 | 11968 | 26,317,632 | 30.05 |
| run1 | bfloat16-32x1024 | wall | eager | 2199 | 200 | 439,800 | 32.64 |
| run1 | bfloat16-32x1024 | wall | compiled | 2199 | 218 | 479,382 | 33.05 |
| run1 | bfloat16-32x1024 | wall | triton_fused | 2199 | 496 | 1,090,704 | 30.01 |
| run1 | bfloat16-32x4096 | graph_gpu | eager | 2184 | 864 | 1,886,976 | 30.71 |
| run1 | bfloat16-32x4096 | graph_gpu | compiled | 2184 | 8992 | 19,638,528 | 30.02 |
| run1 | bfloat16-32x4096 | graph_gpu | triton_fused | 2184 | 8288 | 18,100,992 | 30.06 |
| run1 | bfloat16-32x4096 | wall | eager | 2184 | 195 | 425,880 | 32.46 |
| run1 | bfloat16-32x4096 | wall | compiled | 2184 | 219 | 478,296 | 33.09 |
| run1 | bfloat16-32x4096 | wall | triton_fused | 2184 | 498 | 1,087,632 | 30.67 |
| run1 | bfloat16-512x1024 | graph_gpu | eager | 2193 | 704 | 1,543,872 | 31.82 |
| run1 | bfloat16-512x1024 | graph_gpu | compiled | 2193 | 5632 | 12,350,976 | 30.11 |
| run1 | bfloat16-512x1024 | graph_gpu | triton_fused | 2193 | 6336 | 13,894,848 | 30.03 |
| run1 | bfloat16-512x1024 | wall | eager | 2193 | 197 | 432,021 | 32.63 |
| run1 | bfloat16-512x1024 | wall | compiled | 2193 | 222 | 486,846 | 33.22 |
| run1 | bfloat16-512x1024 | wall | triton_fused | 2193 | 496 | 1,087,728 | 30.42 |
| run1 | bfloat16-512x4096 | graph_gpu | eager | 2325 | 288 | 669,600 | 35.97 |
| run1 | bfloat16-512x4096 | graph_gpu | compiled | 2325 | 2368 | 5,505,600 | 32.56 |
| run1 | bfloat16-512x4096 | graph_gpu | triton_fused | 2325 | 2016 | 4,687,200 | 33.25 |
| run1 | bfloat16-512x4096 | wall | eager | 2325 | 191 | 444,075 | 33.50 |
| run1 | bfloat16-512x4096 | wall | compiled | 2325 | 218 | 506,850 | 34.97 |
| run1 | bfloat16-512x4096 | wall | triton_fused | 2325 | 461 | 1,071,825 | 30.03 |
| run1 | bfloat16-4096x1024 | graph_gpu | eager | 2379 | 96 | 228,384 | 41.65 |
| run1 | bfloat16-4096x1024 | graph_gpu | compiled | 2379 | 416 | 989,664 | 35.34 |
| run1 | bfloat16-4096x1024 | graph_gpu | triton_fused | 2379 | 1568 | 3,730,272 | 33.64 |
| run1 | bfloat16-4096x1024 | wall | eager | 2379 | 79 | 187,941 | 35.28 |
| run1 | bfloat16-4096x1024 | wall | compiled | 2379 | 214 | 509,106 | 35.38 |
| run1 | bfloat16-4096x1024 | wall | triton_fused | 2379 | 441 | 1,049,139 | 30.01 |
| run1 | bfloat16-4096x4096 | graph_gpu | eager | 2220 | 32 | 71,040 | 120.39 |
| run1 | bfloat16-4096x4096 | graph_gpu | compiled | 2220 | 96 | 213,120 | 47.61 |
| run1 | bfloat16-4096x4096 | graph_gpu | triton_fused | 2220 | 96 | 213,120 | 44.49 |
| run1 | bfloat16-4096x4096 | wall | eager | 2220 | 9 | 19,980 | 34.55 |
| run1 | bfloat16-4096x4096 | wall | compiled | 2220 | 62 | 137,640 | 33.52 |
| run1 | bfloat16-4096x4096 | wall | triton_fused | 2220 | 62 | 137,640 | 30.01 |
| run1 | float32-1x1024 | graph_gpu | eager | 2208 | 2208 | 4,875,264 | 30.50 |
| run1 | float32-1x1024 | graph_gpu | compiled | 2208 | 13920 | 30,735,360 | 30.03 |
| run1 | float32-1x1024 | graph_gpu | triton_fused | 2208 | 12768 | 28,191,744 | 30.18 |
| run1 | float32-1x1024 | wall | eager | 2208 | 293 | 646,944 | 32.51 |
| run1 | float32-1x1024 | wall | compiled | 2208 | 211 | 465,888 | 32.12 |
| run1 | float32-1x1024 | wall | triton_fused | 2208 | 498 | 1,099,584 | 30.35 |
| run1 | float32-1x4096 | graph_gpu | eager | 2187 | 2080 | 4,548,960 | 30.34 |
| run1 | float32-1x4096 | graph_gpu | compiled | 2187 | 8672 | 18,965,664 | 30.14 |
| run1 | float32-1x4096 | graph_gpu | triton_fused | 2187 | 8608 | 18,825,696 | 30.03 |
| run1 | float32-1x4096 | wall | eager | 2187 | 294 | 642,978 | 32.18 |
| run1 | float32-1x4096 | wall | compiled | 2187 | 216 | 472,392 | 32.92 |
| run1 | float32-1x4096 | wall | triton_fused | 2187 | 502 | 1,097,874 | 30.11 |
| run1 | float32-32x1024 | graph_gpu | eager | 2199 | 1728 | 3,799,872 | 30.14 |
| run1 | float32-32x1024 | graph_gpu | compiled | 2199 | 12288 | 27,021,312 | 30.04 |
| run1 | float32-32x1024 | graph_gpu | triton_fused | 2199 | 11232 | 24,699,168 | 30.16 |
| run1 | float32-32x1024 | wall | eager | 2199 | 289 | 635,511 | 32.36 |
| run1 | float32-32x1024 | wall | compiled | 2199 | 218 | 479,382 | 32.79 |
| run1 | float32-32x1024 | wall | triton_fused | 2199 | 512 | 1,125,888 | 30.88 |
| run1 | float32-32x4096 | graph_gpu | eager | 2184 | 1152 | 2,515,968 | 30.19 |
| run1 | float32-32x4096 | graph_gpu | compiled | 2184 | 7104 | 15,515,136 | 30.01 |
| run1 | float32-32x4096 | graph_gpu | triton_fused | 2184 | 6976 | 15,235,584 | 30.09 |
| run1 | float32-32x4096 | wall | eager | 2184 | 286 | 624,624 | 32.44 |
| run1 | float32-32x4096 | wall | compiled | 2184 | 216 | 471,744 | 32.86 |
| run1 | float32-32x4096 | wall | triton_fused | 2184 | 497 | 1,085,448 | 30.42 |
| run1 | float32-512x1024 | graph_gpu | eager | 2298 | 992 | 2,279,616 | 34.83 |
| run1 | float32-512x1024 | graph_gpu | compiled | 2298 | 3872 | 8,897,856 | 31.75 |
| run1 | float32-512x1024 | graph_gpu | triton_fused | 2298 | 3776 | 8,677,248 | 31.78 |
| run1 | float32-512x1024 | wall | eager | 2298 | 287 | 659,526 | 33.50 |
| run1 | float32-512x1024 | wall | compiled | 2298 | 211 | 484,878 | 33.30 |
| run1 | float32-512x1024 | wall | triton_fused | 2298 | 479 | 1,100,742 | 30.03 |
| run1 | float32-512x4096 | graph_gpu | eager | 2481 | 384 | 952,704 | 40.39 |
| run1 | float32-512x4096 | graph_gpu | compiled | 2481 | 1088 | 2,699,328 | 34.85 |
| run1 | float32-512x4096 | graph_gpu | triton_fused | 2481 | 1152 | 2,858,112 | 35.16 |
| run1 | float32-512x4096 | wall | eager | 2481 | 271 | 672,351 | 36.58 |
| run1 | float32-512x4096 | wall | compiled | 2481 | 218 | 540,858 | 37.36 |
| run1 | float32-512x4096 | wall | triton_fused | 2481 | 427 | 1,059,387 | 30.01 |
| run1 | float32-4096x1024 | graph_gpu | eager | 2190 | 96 | 210,240 | 39.07 |
| run1 | float32-4096x1024 | graph_gpu | compiled | 2190 | 160 | 350,400 | 34.93 |
| run1 | float32-4096x1024 | graph_gpu | triton_fused | 2190 | 160 | 350,400 | 38.83 |
| run1 | float32-4096x1024 | wall | eager | 2190 | 77 | 168,630 | 32.76 |
| run1 | float32-4096x1024 | wall | compiled | 2190 | 130 | 284,700 | 32.32 |
| run1 | float32-4096x1024 | wall | triton_fused | 2190 | 115 | 251,850 | 30.04 |
| run1 | float32-4096x4096 | graph_gpu | eager | 2064 | 32 | 66,048 | 97.28 |
| run1 | float32-4096x4096 | graph_gpu | compiled | 2064 | 64 | 132,096 | 59.84 |
| run1 | float32-4096x4096 | graph_gpu | triton_fused | 2064 | 64 | 132,096 | 57.43 |
| run1 | float32-4096x4096 | wall | eager | 2064 | 10 | 20,640 | 30.96 |
| run1 | float32-4096x4096 | wall | compiled | 2064 | 33 | 68,112 | 32.12 |
| run1 | float32-4096x4096 | wall | triton_fused | 2064 | 32 | 66,048 | 30.04 |

## Profile evidence

Separate profiling of one warmed call per case (outside timed rounds):

| Shape / dtype | Method | GPU kernels |
|---|---|---:|
| 1 × 4096 / bfloat16 | eager | 10 |
| 1 × 4096 / bfloat16 | compiled | 1 |
| 1 × 4096 / bfloat16 | triton_fused | 1 |
| 4096 × 4096 / bfloat16 | eager | 10 |
| 4096 × 4096 / bfloat16 | compiled | 1 |
| 4096 × 4096 / bfloat16 | triton_fused | 1 |

[Kernel names and counts](profile.json). This supports the launch-fusion explanation; no hardware-counter measurement of DRAM traffic was performed.

## Interpretation and limitations

These measurements establish a local baseline for later custom kernels. The graph result isolates device execution more closely; the Python result also reflects dispatch, allocation and synchronization overhead. Neither is end-to-end model throughput. Reused inputs may fit in GPU cache for small shapes; do not extrapolate these ratios to streaming-memory workloads.

A sustained run does not establish cross-day or cross-machine reproducibility. Multiple process runs, when present, check immediate repeatability on the same machine. Windows display activity, thermal state, power management, and unlocked clocks can affect results. Per-case GPU snapshots are retained in summaries. Do not average the two timing modes together.

## Instrumentation and observer effects

Timer reads, duration-floor decisions, sample writes, and progress reporting occur outside the timed batches. They add total run time and gaps between batches. External SSH status reads can still contend for host CPU, and driver queries can disturb scheduling or power state; those indirect effects are not quantified here. Treat the Python wall-clock results as more exposed to host activity. See the exercise notes for how a published run was monitored. This is not a controlled comparison against a completely unobserved run.

## Reproduction and later optimizations

From the exercise directory on the GPU host:

```bash
source env.sh
python src/benchmark.py --config configs/long.json --methods eager compiled triton_fused --out local/new-run1
python src/benchmark.py --config configs/long.json --methods eager compiled triton_fused --out local/new-run2
python analysis/report.py local/new-run1 local/new-run2 --out local/new-report
```

Output directories must be new. `COMPLETE` is written only after all cases pass. Raw numeric samples stay in ignored `local/`; reviewed manifests, summaries, CSV and charts can be published. Source/configuration SHA256 values identify the measured experiment. Add later kernels through `implementations.build`, keep the operation and protocol fixed, and include both eager and compiled baselines in each new run.
