# Qwen3-235B across two A100 80GB GPUs

**Queued follow-up. Do not start before the
[disaggregated-serving exercise](../07_disaggregated_serving/) is complete.**
No model download, GPU allocation, serving test, or benchmark has been performed
for this exercise.

Host one quantized Qwen3-235B-A22B model across two A100-SXM4-80GB GPUs and compare
parallelism modes that fit its architecture and serving backend. The candidate is
[AIDXteam/Qwen3-235B-A22B-Thinking-2507-AWQ](https://huggingface.co/AIDXteam/Qwen3-235B-A22B-Thinking-2507-AWQ).
Its safetensors index declares 123,142,620,032 bytes of tensors (123.14 GB;
114.69 GiB). This is an initial storage/memory feasibility check, not a verified
runtime footprint. KV cache, quantization workspaces, CUDA graphs, and per-rank
imbalance must still fit. Pin and review the checkpoint revision before execution.

## Comparisons to validate

| Mode | Proposed arrangement | Question |
| --- | --- | --- |
| Tensor parallel | TP=2, PP=1 | Can the quantized model serve reliably, and at what latency and throughput? |
| Pipeline parallel | TP=1, PP=2 | Does splitting layers improve or worsen the balance of memory, communication, and pipeline idle time? |
| Expert parallel | Two expert ranks, with a compatible attention-parallel arrangement | Does sharding MoE experts perform better than tensor-sharding their weights? |

Expert-parallel support for this exact AWQ implementation must be checked first.
If unsupported, record the limitation rather than silently switching checkpoint
or quantization. A different supported quantization would be a separate control.
Qwen3-235B has approximately 22B active parameters per token, but all expert
weights must remain available; active parameter count is not its weight-memory
requirement. Two independent full-model GPU replicas do not fit this checkpoint
on these two devices.

## Execution and evidence

1. Complete exercise 07 and retain its results before starting this study.
2. Pin model, quantization, engine, and container; provision sufficient storage
   for the roughly 123 GB checkpoint plus environment and captures. The current
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

Sources: [checkpoint tensor index](https://huggingface.co/AIDXteam/Qwen3-235B-A22B-Thinking-2507-AWQ/blob/main/model.safetensors.index.json),
[Qwen3-235B model architecture](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507),
[vLLM parallelism](https://docs.vllm.ai/en/v0.24.0/serving/parallelism_scaling/).
