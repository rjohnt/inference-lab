# Qwen3-235B across two A100 80GB GPUs

**Authorized follow-up to the [disaggregated-serving exercise](../07_disaggregated_serving/).**
The workflow waits for the preceding measurements and verified archive, then
runs this study on the same two-GPU rental. The checkpoint is downloaded and
the corrected TP pilot passed. The full sweep is running; results will distinguish
successful modes from failed attempts.

Host one quantized Qwen3-235B-A22B model across two A100-SXM4-80GB GPUs and compare
parallelism modes that fit its architecture and serving backend. The candidate is
[QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ](https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ).
Its safetensors index declares 124,055,313,408 bytes of tensors (124.06 GB;
115.54 GiB). This is an initial storage/memory feasibility check, not a verified
runtime footprint. KV cache, quantization workspaces, CUDA graphs, and per-rank
imbalance must still fit. The candidate revision is pinned to
`38e8e75808020b2c9be573c19629a4d02cd8e8aa`. The TP pilot loaded the model with
57.92 GiB of reported model memory per GPU using Marlin AWQ/MoE kernels and
FlashAttention. Full-context TP, PP, and EP measurements are still pending.

## Why this candidate

The checkpoint has 235B total parameters, 22B active per token, 94 layers and
128 experts. Its AWQ configuration uses 4-bit weights, 128-element groups and
asymmetric zero points. The tensor index reports about 57.77 GiB per GPU if
perfectly balanced, before runtime allocations. Start with a 9,216-token context limit (room for an 8K prompt plus output)
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
comparison. Warm each configuration before three measured repetitions. Run a small TP pilot first, then test concurrency 1, 4, and 16 on
three workload shapes: up to 1K input/1K output, 8K input/512 output, and
32K input/128 output. Full runs use a 33,792-token context limit, 16 maximum
sequences, a 4K batch budget, and 90% GPU-memory utilization. Each measured
run submits `max(16, 4 * concurrency)` requests; one separate warmup run
precedes three measured repetitions per cell.

Report output tokens/s, completed requests/s, TTFT, TPOT, streaming inter-token
latency, end-to-end latency, errors, and per-GPU memory/utilization. Record any
latency targets explicitly when reporting goodput. Keep forced-length performance
runs separate from answer-quality checks, and count generated reasoning tokens
consistently across modes. Pin the CLI version and capture its effective options;
retain raw benchmark JSON privately and publish reviewed summaries and charts.
These are custom Qwen serving measurements, not official MLPerf results.
[Benchmark documentation](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md).

The private input builder selects documents from LongBench's `qasper`,
`gov_report`, and `hotpotqa` tasks at pinned revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. It truncates context to the target
budget while preserving a task instruction and the model's chat template. It
records actual token lengths, source counts, and file hashes. These modified,
forced-length prompts are serving workloads, not a scored LongBench evaluation.
The source pool is repeated if fewer than 64 eligible documents are available;
prefix caching is disabled in every mode. CLI seed 42 fixes request shuffling.
The prepared inputs contain 968, 8,136, and 32,712 tokens respectively. The
first two shapes use 64 distinct source documents each; the longest shape has
only three eligible documents, repeated to fill the 64-request pool. That limits
its content diversity; it is a controlled long-context stress test.

## Execution and evidence

1. Complete exercise 07 and retain its results before starting this study.
2. Pin model, quantization, engine, and container; provision sufficient storage
   for the roughly 124 GB checkpoint plus environment and captures. This run uses
   a shared-memory filesystem with 233 GiB capacity for weights; captures remain
   on persistent storage and are copied locally.
3. Verify actual NVLink topology, peer access, per-rank allocation, and supported
   parallelism before starting timed work. Retrieve and verify artifacts, then
   publish reviewed results before teardown.
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

## Run the prepared workflow

After the preceding study is archived, provide enough storage for the
124 GB checkpoint, environment, and private captures. This run uses a symlink
from the model directory to a shared-memory filesystem (233 GiB capacity),
avoiding slow persistent-volume reads. Weights are disposable and pinned for
redownload; all result captures remain on persistent storage. Copy `src/` to
`/workspace/qwen235b/`. These scripts use the recorded vLLM environment at
`/opt/disagg/venv`; install the environment before running on a fresh node.

```bash
uv pip install --python /opt/disagg/venv/bin/python 'vllm[bench]==0.24.0'
cd /workspace/qwen235b
bash experiment.sh
```

The workflow downloads the pinned checkpoint, prepares private inputs, verifies
the TP pilot, then runs TP, PP and EP in separate server processes. Each phase
retains its exit status and logs. A failed full phase does not prevent trying
the remaining modes. `EXPERIMENT_COMPLETE` is written only if all three succeed.
Keep the rental until raw archives are verified and reviewed results are pushed.

Import completed modes locally with:

```bash
python3 analysis/summarize.py /path/to/private/raw
```

The initial TP launch loaded the model and passed four 64-token smoke checks,
but the first benchmark client exited before timed requests because the
benchmark extras were missing. Installing `vllm[bench]` preserved vLLM 0.24.0
and PyTorch 2.11.0. During setup review, the command was also corrected to use
`--custom-output-len -1`: this CLI otherwise overrides dataset output budgets
with its 256-token default. The corrected harness asserts both completed
request count and total output-token count for every run. Failed setup logs
and the initial GPU trace are retained privately; they are excluded from
measured serving results.
