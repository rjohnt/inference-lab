# Curriculum roadmap

## Strategy

This is a sequence of engineering investigations, ordered from reliable
measurement to full-system inference optimization. A stage is complete only when
its experiment is runnable, its inputs and environment are recorded, its results
are plotted or tabulated, and the result is explained in a report. The work should
remain useful on an RTX 3080-class machine; higher-end or multi-GPU hardware is
reserved for comparisons that genuinely require it.

The durable portfolio unit is a performance study:

```text
Question → prediction → controlled experiment → measurement → profile
        → bottleneck hypothesis → change → validation → engineering conclusion
```

## Stages

| Stage | Question to answer | Primary evidence | Compute tier |
| --- | --- | --- | --- |
| 00. Measurement foundations | Can the harness measure GPU work correctly and repeatably? | Latency/throughput curves for matmul, softmax, normalization, and copies across dtype and size | Local GPU |
| 01. Runtime and data movement | When is apparent GPU slowness caused by the host/runtime? | Pinned vs. pageable, sync vs. async, and CPU-starvation studies | Local GPU |
| 02. CUDA execution and memory | How do launch geometry and memory access affect kernel behavior? | Block-size sweeps plus uncoalesced-before/after profile study | Local GPU |
| 03. PyTorch/transformer execution | What dominates a small transformer's runtime, and which framework-level knobs help? | Profiler trace and eager/compile/dtype/attention comparison | Local GPU |
| 04. Triton and fusion | When does a custom fused kernel beat a framework composition? | PyTorch-versus-Triton sweeps for softmax/RMSNorm and one fused operation | Local GPU |
| 05. LLM serving | How do prefill, decode, and continuous batching behave under load? | TTFT, TPOT, tail latency, throughput, utilization, and saturation curves | Local GPU |
| 06. Capacity and memory | Can KV-cache use be predicted, then measured, under context/concurrency pressure? | Predicted-versus-measured memory model plus quantization trade-off | Local GPU + optional high-end GPU |
| 07. Production operations | Which system limit breaks the service SLO first, and why? | Metrics dashboard, overload test, queueing analysis, and remediation | Local GPU |
| 08. Distributed inference | Why does adding GPUs not scale linearly? | 1→2 (optional 4) GPU efficiency and communication analysis | Rented multi-GPU |
| 09. Capstone | How should an inference service be designed, benchmarked, and optimized for a target SLO and cost? | End-to-end baseline/profile/optimization/cost case study | Local + representative high-end GPU |

## Evidence ladder

Each stage should add a different kind of evidence rather than repeating the same
benchmark at a larger scale.

1. **Correctness:** the operation and measurement procedure are validated.
2. **Characterization:** performance is mapped over the variable that matters.
3. **Diagnosis:** a profiler or system metric identifies the limiting resource.
4. **Intervention:** a pre-stated change is tested against a baseline.
5. **Explanation:** the result is tied to GPU/runtime/serving behavior and its
   limitations are stated.
6. **Operational judgment:** a recommendation identifies the best configuration
   for a stated latency, throughput, memory, or cost objective.

## Hardware policy

- Develop and iterate locally, with a Linux NVIDIA workstation as the execution
  environment and the Mac as the control/development machine.
- Record GPU model, driver, CUDA runtime, framework versions, power policy, and
  model revision in every report.
- Use a single local GPU for foundational and most serving experiments.
- Use rented high-end hardware only for a representative cross-hardware comparison
  and distributed experiments; reproduce the smallest decisive workload there.
- Avoid treating results from two hardware classes as directly comparable without
  normalizing the workload and documenting the configuration.

## Recommended cadence

Treat the roadmap as roughly twelve weeks of focused work, but advance by evidence
rather than calendar. A useful weekly rhythm is:

1. Frame one question and prediction.
2. Implement a minimal, repeatable benchmark.
3. Run a small pilot, validate measurement, then run the full sweep.
4. Profile the surprising or representative case.
5. Publish code, compact results, and a one-page conclusion.

Do not start the next stage with an unfinished report from the current one.

## Capstone acceptance criteria

The capstone should make a reader able to reproduce the decision process:

- a workload definition and explicit SLO;
- baseline TTFT, TPOT, throughput, p50/p95/p99, VRAM, utilization, and cost model;
- a profile-backed bottleneck hypothesis;
- several independently measured optimization attempts, including at least one
  unsuccessful or neutral result;
- a recommended configuration, trade-offs, and constraints;
- a representative comparison between local hardware and a high-end GPU.
