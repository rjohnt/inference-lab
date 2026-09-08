# Speculative decoding benchmark

**Partial run: do not treat these results as final.**

Same Qwen3-8B Q4_K_M target, 32K context, q8_0 KV cache, GPU layers all, Flash Attention on, thinking on, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | none | 1 | 0.06 | 1.78 | 2.51 | 1.00× | 79.51 | — | 0 |
| chat_short | 12 | ngram-mod | 1 | 0.06 | 2.04 | 2.69 | 0.87× | 78.87 | 0% | 0 |
| code_512 | 634 | none | 1 | 0.17 | 7.16 | 8.46 | 1.00× | 77.34 | — | 0 |
| code_512 | 634 | ngram-mod | 1 | 0.19 | 6.86 | 7.18 | 1.04× | 93.83 | 86% | 0 |
| code_8192 | 8314 | none | 1 | 1.92 | 10.43 | 11.91 | 1.00× | 67.87 | — | 0 |
| code_8192 | 8314 | ngram-mod | 1 | 1.91 | 9.67 | 9.98 | 1.08× | 109.21 | 69% | 0 |
| document_2048 | 2115 | none | 1 | 0.47 | 3.68 | 4.12 | 1.00× | 75.61 | — | 0 |
| document_2048 | 2115 | ngram-mod | 1 | 0.46 | 3.67 | 11.00 | 1.00× | 31.32 | 45% | 0 |
| document_512 | 578 | none | 1 | 0.17 | 2.45 | 2.88 | 1.00× | 74.26 | — | 0 |
| document_512 | 578 | ngram-mod | 1 | 0.17 | 2.48 | 2.91 | 0.98× | 82.90 | 53% | 0 |
| document_8192 | 8258 | none | 1 | 1.94 | 5.22 | 5.86 | 1.00× | 63.44 | — | 0 |
| document_8192 | 8258 | ngram-mod | 1 | 1.89 | 5.02 | 5.60 | 1.04× | 67.15 | — | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.

- chat_short, ngram-mod: 0/1 exact outputs match baseline.
- code_512, ngram-mod: 0/1 exact outputs match baseline.
- code_8192, ngram-mod: 0/1 exact outputs match baseline.
- document_2048, ngram-mod: 0/1 exact outputs match baseline.
- document_512, ngram-mod: 0/1 exact outputs match baseline.
- document_8192, ngram-mod: 1/1 exact outputs match baseline.

Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: F16 RedHatAI-derived GGUF, max draft=3. These are initial configurations, not a search for the optimal draft length.
