# Speculative decoding benchmark

Complete run.

Same Qwen3-8B Q4_K_M target, 32768-token context, q8_0 target KV cache (draft cache settings in command files), GPU layers all, Flash Attention on, thinking off, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | none | 10 | 0.03 | 0.03 | 0.71 | 1.00× | 79.68 | — | 0 |
| chat_short | 12 | draft-eagle3 | 10 | 0.03 | 0.03 | 0.52 | 0.91× | 103.25 | 39% | 0 |
| chat_short | 12 | draft-dflash | 10 | 0.03 | 0.03 | 0.52 | 0.92× | 101.44 | 12% | 0 |
| code_512 | 634 | none | 10 | 0.19 | 0.19 | 1.46 | 1.00× | 79.01 | — | 0 |
| code_512 | 634 | draft-eagle3 | 10 | 0.23 | 0.23 | 0.82 | 0.82× | 168.89 | 84% | 0 |
| code_512 | 634 | draft-dflash | 10 | 0.22 | 0.22 | 0.45 | 0.85× | 435.26 | 79% | 0 |
| code_8192 | 8314 | none | 10 | 2.36 | 2.36 | 3.86 | 1.00× | 67.81 | — | 0 |
| code_8192 | 8314 | draft-eagle3 | 10 | 2.86 | 2.86 | 3.60 | 0.83× | 136.22 | 80% | 0 |
| code_8192 | 8314 | draft-dflash | 10 | 2.75 | 2.75 | 3.18 | 0.86× | 238.45 | 46% | 0 |
| document_2048 | 2115 | none | 10 | 0.55 | 0.55 | 1.06 | 1.00× | 75.50 | — | 0 |
| document_2048 | 2115 | draft-eagle3 | 10 | 0.67 | 0.67 | 0.94 | 0.82× | 146.70 | 78% | 0 |
| document_2048 | 2115 | draft-dflash | 10 | 0.65 | 0.65 | 0.80 | 0.84× | 260.04 | 48% | 0 |
| document_512 | 578 | none | 10 | 0.16 | 0.16 | 0.66 | 1.00× | 78.53 | — | 0 |
| document_512 | 578 | draft-eagle3 | 10 | 0.20 | 0.20 | 0.45 | 0.80× | 154.52 | 78% | 0 |
| document_512 | 578 | draft-dflash | 10 | 0.19 | 0.19 | 0.34 | 0.85× | 272.31 | 48% | 0 |
| document_8192 | 8258 | none | 10 | 2.33 | 2.33 | 2.82 | 1.00× | 67.29 | — | 0 |
| document_8192 | 8258 | draft-eagle3 | 10 | 2.85 | 2.85 | 3.17 | 0.82× | 101.15 | 56% | 0 |
| document_8192 | 8258 | draft-dflash | 10 | 2.75 | 2.75 | 3.04 | 0.85× | 114.06 | 19% | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.

- chat_short, draft-eagle3: 0/10 exact outputs match baseline.
- chat_short, draft-dflash: 0/10 exact outputs match baseline.
- code_512, draft-eagle3: 10/10 exact outputs match baseline.
- code_512, draft-dflash: 10/10 exact outputs match baseline.
- code_8192, draft-eagle3: 10/10 exact outputs match baseline.
- code_8192, draft-dflash: 10/10 exact outputs match baseline.
- document_2048, draft-eagle3: 10/10 exact outputs match baseline.
- document_2048, draft-dflash: 10/10 exact outputs match baseline.
- document_512, draft-eagle3: 10/10 exact outputs match baseline.
- document_512, draft-dflash: 10/10 exact outputs match baseline.
- document_8192, draft-eagle3: 10/10 exact outputs match baseline.
- document_8192, draft-dflash: 10/10 exact outputs match baseline.

Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.
