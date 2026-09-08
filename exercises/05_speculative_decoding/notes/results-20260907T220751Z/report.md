# Speculative decoding benchmark

**Partial run: do not treat these results as final.**

Same Qwen3-8B Q4_K_M target, 32768-token context, q8_0 KV cache, GPU layers all, Flash Attention on, thinking on, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | none | 1 | 0.06 | 1.77 | 2.49 | 1.00× | 79.91 | — | 0 |
| chat_short | 12 | ngram-mod | 1 | 0.06 | 2.03 | 2.68 | 0.87× | 79.13 | 0% | 0 |
| code_512 | 634 | none | 1 | 0.17 | 7.11 | 8.40 | 1.00× | 77.84 | — | 0 |
| code_512 | 634 | ngram-mod | 1 | 0.18 | 6.82 | 7.14 | 1.04× | 94.10 | 86% | 0 |
| code_8192 | 8314 | none | 1 | 1.92 | 10.47 | 11.97 | 1.00× | 67.46 | — | 0 |
| code_8192 | 8314 | ngram-mod | 1 | 1.92 | 9.67 | 9.99 | 1.08× | 109.18 | 69% | 0 |
| document_2048 | 2115 | none | 1 | 0.47 | 3.67 | 4.11 | 1.00× | 75.78 | — | 0 |
| document_2048 | 2115 | ngram-mod | 1 | 0.46 | 3.68 | 4.13 | 1.00× | 89.60 | 45% | 0 |
| document_512 | 578 | none | 1 | 0.16 | 2.30 | 2.72 | 1.00× | 78.76 | — | 0 |
| document_512 | 578 | ngram-mod | 1 | 0.17 | 2.47 | 2.90 | 0.93× | 83.28 | 53% | 0 |
| document_8192 | 8258 | none | 1 | 1.89 | 4.99 | 5.56 | 1.00× | 67.69 | — | 0 |
| document_8192 | 8258 | ngram-mod | 1 | 1.89 | 5.01 | 5.59 | 1.00× | 67.29 | — | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.

- chat_short, ngram-mod: 0/1 exact outputs match baseline.
- code_512, ngram-mod: 0/1 exact outputs match baseline.
- code_8192, ngram-mod: 0/1 exact outputs match baseline.
- document_2048, ngram-mod: 0/1 exact outputs match baseline.
- document_512, ngram-mod: 0/1 exact outputs match baseline.
- document_8192, ngram-mod: 1/1 exact outputs match baseline.

Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.
