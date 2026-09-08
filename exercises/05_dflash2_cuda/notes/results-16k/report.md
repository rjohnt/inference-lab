# Qwen3.8-27B UD-IQ2_S: baseline versus DFlash 2

120 measured requests; 10 repetitions per cell; thinking off; context 16,384.

Exact same six prompt texts as the previous Qwen3-8B run, retokenized for this target. Method order alternates by repetition. Warm, uncached requests; greedy sampling, seed 42, 1536-token output cap; Q8 KV; all model layers requested on GPU. Raw logs record actual placement.

Quality: 0 retrieval failures; 40 code outputs need reference review; 0 truncated.
Code review means the AST differs from the reference; this alone does not establish functional incorrectness.

| Case | Baseline tok/s | DFlash 2 tok/s | Decode ratio | Baseline total s | DFlash total s | Total-time speedup | Exact matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 42.90 | 55.71 | 1.30× | 2.223 | 1.893 | 1.17× | 10/10 |
| document_512 | 42.51 | 122.23 | 2.88× | 1.968 | 1.491 | 1.32× | 10/10 |
| document_2048 | 42.12 | 121.89 | 2.89× | 3.513 | 3.137 | 1.12× | 10/10 |
| document_8192 | 40.10 | 110.47 | 2.75× | 9.084 | 9.443 | 0.96× | 10/10 |
| code_512 | 42.64 | 132.54 | 3.11× | 3.240 | 1.968 | 1.65× | 10/10 |
| code_8192 | 40.55 | 134.94 | 3.33× | 10.939 | 10.221 | 1.07× | 10/10 |

Decode throughput excludes prompt prefill; total time includes it. This measures the IQ2_S target and Q4_K_M drafter on this PC; it is not a direct algorithm comparison with the earlier Qwen3-8B DFlash 1 results.

Model revisions, SHA256 checksums, server version and settings: config.json. Per-request output and metrics: results.jsonl. Quality checks: validation.json.

Code review completed: all 40 code responses share one AST and are equivalent to the reference for numeric-list inputs by source inspection. See [code-review.md](code-review.md).
