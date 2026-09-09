# Qwen3-235B across two A100 80GB GPUs

**Queued follow-up. Do not start before the
[disaggregated-serving exercise](../07_disaggregated_serving/) is complete.**
No model download, GPU allocation, serving test, or benchmark has been performed
for this exercise.

Host one quantized Qwen3-235B-A22B model across two A100-SXM4-80GB GPUs and compare
parallelism modes that fit its architecture and serving backend. The candidate is
[QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ](https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ).
Its safetensors index declares 124,055,313,408 bytes of tensors (124.06 GB;
115.54 GiB). This is an initial storage/memory feasibility check, not a verified
runtime footprint. KV cache, quantization workspaces, CUDA graphs, and per-rank
imbalance must still fit. The candidate revision is pinned to
`38e8e75808020b2c9be573c19629a4d02cd8e8aa`; runtime compatibility is still unverified.

## Why this candidate

The checkpoint has 235B total parameters, 22B active per token, 94 layers and
128 experts. Its AWQ configuration uses 4-bit weights, 128-element groups and
asymmetric zero points. The tensor index reports about 57.77 GiB per GPU if
perfectly balanced, before runtime allocations. Start with an 8K context limit
and four sequences; measure each rank's real memory before expanding either.

vLLM 0.24 documents Qwen3MoE pipeline support, and this checkpoint's publisher
provides an expert-parallel vLLM example. Inspection of the installed Marlin MoE
backend confirms INT4 support and no general rejection of expert parallelism;
it does reject specific FlashInfer NVLink backends. These are compatibility
signals, not a successful two-A100 serving test. Select and verify a compatible
collective backend during execution.

The initially considered AIDX checkpoint is a distinct `compressed-tensors`
INT4 representation despite its AWQ repository name (123.14 GB of tensors).
The QuantTrio checkpoint provides standard AWQ metadata and an explicit EP
example, making it the first candidate for this parallelism comparison.
MiniMax-M2.5 AWQ is an alternative near 229B parameters, but its QuantTrio tensor
index is larger at 130.21 GB; it does not improve our parameter-capacity target.
A hypothetical 300B model already requires 150 GB at exactly four bits per
parameter before scales, unquantized tensors, KV cache or runtime workspaces.
That is a tighter capacity experiment, not an established drop-in alternative.

## Comparisons to validate

| Mode | Proposed arrangement | Question |
| --- | --- | --- |
| Tensor parallel | TP=2, PP=1 | Can the quantized model serve reliably, and at what latency and throughput? |
| Pipeline parallel | TP=1, PP=2 | Does splitting layers improve or worsen the balance of memory, communication, and pipeline idle time? |
| Tensor + expert parallel | TP=2 with EP enabled | Compare expert sharding against TP-sharded experts, keeping attention TP=2 |
| Data + expert parallel (optional) | Attention DP=2, TP=1; experts EP=2 | Compare replicated attention with partitioned experts if memory and collectives permit |

Expert-parallel support for this exact AWQ implementation must be checked first.
If unsupported, record the limitation rather than silently switching checkpoint
or quantization. A different supported quantization would be a separate control.
Qwen3-235B has approximately 22B active parameters per token, but all expert
weights must remain available; active parameter count is not its weight-memory
requirement. Two independent full-model GPU replicas do not fit this checkpoint
on these two devices.

## Benchmark harness

Use **`vllm bench serve`**, with **TP=2** as the baseline for PP=2 and
TP=2 with expert parallel enabled. Keep the checkpoint, tokenizer, request
sample/order, input/output budgets, and offered load fixed in each paired
comparison. Warm each configuration before three measured repetitions. Finalize
the workload sizes and concurrency sweep after the capacity pilot; include
short-input/long-output, long-input/short-output, and intermediate workloads.

Report output tokens/s, completed requests/s, TTFT, TPOT, streaming inter-token
latency, end-to-end latency, errors, and per-GPU memory/utilization. Record any
latency targets explicitly when reporting goodput. Keep forced-length performance
runs separate from answer-quality checks, and count generated reasoning tokens
consistently across modes. Pin the CLI version and capture its effective options;
retain raw benchmark JSON privately and publish reviewed summaries and charts.
These are custom Qwen serving measurements, not official MLPerf results.
[Benchmark documentation](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md).

## Execution and evidence

1. Complete exercise 07 and retain its results before starting this study.
2. Pin model, quantization, engine, and container; provision sufficient storage
   for the roughly 124 GB checkpoint plus environment and captures. The current
   exercise's storage allocation is not a suitable assumption for this model.
3. Verify actual NVLink topology, peer access, per-rank allocation, and supported
   parallelism before starting timed work. Use a rental deadline and retrieve
   artifacts before teardown.
4. Check fixed greedy outputs across supported modes, retaining any differences.
5. Compare the same prompt/output lengths and concurrency at equal GPU cost.
   Report TTFT, output-token latency, throughput, errors, and per-GPU memory and
   activity. Separate cold load/compile from warmed serving.
6. Publish reviewed results, actual-test diagrams, complete commands/configs,
   and limitations here. Preserve raw captures privately and terminate rentals.

Sources: [checkpoint tensor index](https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ/blob/main/model.safetensors.index.json),
[Qwen3-235B model architecture](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507),
[vLLM parallelism](https://docs.vllm.ai/en/v0.24.0/serving/parallelism_scaling/).

Additional sources: [vLLM supported models](https://docs.vllm.ai/en/v0.24.0/models/supported_models/),
[Ampere quantization support](https://docs.vllm.ai/en/stable/features/quantization/),
[expert parallel deployment](https://docs.vllm.ai/en/v0.24.0/serving/expert_parallel_deployment/),
[MiniMax AWQ tensor index](https://huggingface.co/QuantTrio/MiniMax-M2.5-AWQ/blob/main/model.safetensors.index.json).
