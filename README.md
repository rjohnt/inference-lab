# Inference Lab

A public, evidence-first curriculum for LLM inference and performance engineering.

This repository records reproducible experiments across the inference stack: GPU
fundamentals, PyTorch execution, custom kernels, model serving, capacity planning,
and distributed inference. The goal is to build engineering judgment, not merely
collect implementations. Every completed exercise should answer a concrete
performance question with measurements and an explanation.

## Scope of the two labs

| Repository | Primary scope |
| --- | --- |
| [inference-lab](https://github.com/rjohnt/inference-lab) | Forward-only execution, inference kernel optimization, host/device data movement, model serving, decoding, latency and throughput. |
| [training-lab](https://github.com/rjohnt/training-lab) | Forward-plus-backward execution, gradients, saved activations, training memory, recomputation/checkpointing, optimizer costs and training throughput. |

Choose the home by the main question being measured, rather than the operator's
name. GPU tooling and kernels can support both labs. Keep a study in one primary
home and cross-link it; include clearly labeled controls from the other scope
when useful. The LayerNorm vs RMSNorm memory study belongs in training-lab because
it examines saved activations and backward costs, with forward-only inference
controls. The transfer/pipeline study belongs in inference-lab.

## How this repository is organized

| Location | Purpose |
| --- | --- |
| [`docs/roadmap.md`](docs/roadmap.md) | Curriculum stages, outcomes, and hardware plan |
| [`docs/experiment-contract.md`](docs/experiment-contract.md) | Definition of a complete experiment |
| [`SECURITY.md`](SECURITY.md) | Credential handling, scanning, and public-release procedure |
| [`exercises/`](exercises/) | Runnable code and per-exercise instructions |
| [`reports/`](reports/) | Concise, reviewable performance studies |
| [`artifacts/`](artifacts/) | Generated charts and small derived results (not raw profiler captures) |

## Operating principles

- Benchmark correctness and reproducibility before optimization.
- Predict the bottleneck or outcome before running an experiment.
- Change one meaningful variable at a time and preserve the environment details.
- Report distributions and throughput together; a single average is not enough.
- Treat profiling evidence as a prompt for a hypothesis, not as the conclusion.
- Keep raw traces, model weights, credentials, and machine-specific configuration
  out of version control.

See [`SECURITY.md`](SECURITY.md) before adding a service integration, cloud
workload, or a new configuration file.

## Experiment catalog

Each experiment has its own directory containing its code, README, configuration,
and reviewed results. Start with the experiment that interests you:

| Exercise | Status / evidence |
| --- | --- |
| [Ollama streaming baseline](exercises/00_ollama_baseline/) | Cold/warm TTFT and response timing |
| [GPU runtime and data movement](exercises/01_data_movement/) | 105 measured cases; transfer bandwidth, copy batching, RMSNorm/GEMM pipelines and diagrams |
| [Sequential, streams, and batching](exercises/01_compute_streams/) | 54 measured cases; sequential vs four streams vs batching, plus CUDA Graph replay |
| [C++ GPU extension](exercises/03_cpp_extension/) | Small PyTorch extension and CUDA check |
| [Kernel fusion](exercises/04_kernel_fusion/) | Sustained eager/compiled baseline: 270.8M calls, correctness checks, inline charts |
| [LayerNorm vs RMSNorm — training-lab](https://github.com/rjohnt/training-lab/tree/main/exercises/01_layernorm_vs_rmsnorm) | Planned training-memory study, forward-only controls and shared-axis charts |
| [Nsight Compute RMSNorm profiling](exercises/04_ncu_rmsnorm/) | Prepared for a future NVTX-filtered kernel study; not executed |
| [Qwen3 8B speculative decoding](exercises/05_speculative_decoding/) | Baseline, EAGLE-3, DFlash and n-gram experiments |
| [DFlash 2 on CUDA](exercises/05_dflash2_cuda/) | 120 requests, reports, checks and summaries |
| [DFlash 2 on Metal](exercises/05_dflash2_metal/) | Metal run and 240-response combined report |
| [On-demand model router](exercises/05_on_demand_router/) | Authentication, serialization, idle unloading and wake checks |
| [vLLM streaming benchmark](exercises/05_single_node_serving/) | Existing remote-serving harness |

See [import notes](docs/import-notes.md) for provenance and historical-script
limitations. Raw captures and machine-specific settings remain outside Git.

## Progress

| Stage | Status |
| --- | --- |
| 00 — Measurement foundations | Baseline imported |
| 01 — GPU runtime and data movement | Transfer/pipeline and compute-stream/batching studies complete |
| 02 — CUDA execution and memory behavior | Planned |
| 03 — PyTorch and transformer execution | C++ extension imported |
| 04 — Triton kernels and fusion | Eager/compiled baseline complete; custom kernel next |
| 05 — Single-node LLM serving | Multiple experiments imported |
| 06 — Capacity, KV cache, and quantization | Planned |
| 07 — Production operations | Planned |
| 08 — Multi-GPU inference | Planned |
| 09 — Capstone | Planned |
