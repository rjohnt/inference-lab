# 05 — Single-node LLM serving

`vllm_stream_benchmark.py` measures a live OpenAI-compatible vLLM endpoint from
the client. It is deliberately small and dependency-free so it can be re-run
after each server setting change.

Set the endpoint and key in the shell that will run the test:

```bash
export VLLM_BASE_URL='https://<pod-id>-8000.proxy.runpod.net/v1'
export VLLM_API_KEY='...'
python3 exercises/05_single_node_serving/vllm_stream_benchmark.py \
  --label a6000-128k-baseline --runs 10 --warmup-runs 2 --max-tokens 256
```

It records two complementary measures for each streaming request:

- **TTFT**: client request start to the first streamed text token. This includes
  network and queueing time, which is what an interactive user experiences.
- **Generation TPS**: `completion_tokens / (stream finish - first text token)`.
  The final `usage` object supplied by vLLM supplies the token count, so it is
  not a character-based estimate.

The default `--cache-mode cold` adds a unique harmless suffix to each request,
avoiding prefix-cache hits. Use `--cache-mode warm` to intentionally measure
the benefit of repeated context. The script writes timestamped JSON and CSV
files under `artifacts/raw/vllm-bench/`; those outputs are ignored by Git.

## Tuning loop

1. Record a baseline with the exact command above.
2. Change **one** server setting (for example, `--gpu-memory-utilization`,
   `--max-num-seqs`, or a quantization/model choice), restart vLLM, and wait for
   `/v1/models` to respond.
3. Re-run the same command with a new `--label`, such as `a6000-mem92`.
4. Compare p50 and p95 TTFT, then p50 and p95 generation TPS. Keep the JSON
   files as the evidence for the decision.

Each run performs its own warm-up, so no manual benchmark reset is necessary.
To measure a genuinely cold model start, restart the vLLM process/pod, wait for
the endpoint to become ready, then run the same command with `--warmup-runs 0`
and a fresh label. Treat that separately from steady-state results.
