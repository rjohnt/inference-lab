# Speculative decoding benchmark

**Correction: this run used DFlash 1, not DFlash 2. The upstream Qwen3-8B-DFlash-b16 draft is documented for thinking disabled, but this run enabled thinking for all methods. Results describe that tested configuration and should not be treated as a comparison of optimally configured EAGLE and DFlash. This mismatch may contribute to low draft acceptance; its effect has not been isolated.**

Source: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16

Complete run.

Same Qwen3-8B Q4_K_M target, 32768-token context, q8_0 target KV cache (draft cache settings in command files), GPU layers all, Flash Attention on, thinking on, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | none | 3 | 0.07 | 2.22 | 3.13 | 1.00× | 63.59 | — | 0 |
| chat_short | 12 | draft-eagle3 | 3 | 0.06 | 1.59 | 2.15 | 1.40× | 90.55 | 41% | 0 |
| chat_short | 12 | draft-dflash | 3 | 0.11 | 2.41 | 3.13 | 0.92× | 67.98 | 8% | 0 |
| code_512 | 634 | none | 3 | 0.25 | 12.89 | 14.50 | 1.00× | 61.09 | — | 0 |
| code_512 | 634 | draft-eagle3 | 3 | 0.28 | 7.53 | 8.19 | 1.71× | 100.52 | 51% | 0 |
| code_512 | 634 | draft-dflash | 3 | 0.28 | 8.34 | 8.58 | 1.55× | 101.42 | 15% | 0 |
| code_8192 | 8314 | none | 3 | 2.81 | 13.16 | 15.00 | 1.00× | 54.60 | — | 0 |
| code_8192 | 8314 | draft-eagle3 | 3 | 3.42 | 11.43 | 12.28 | 1.15× | 75.94 | 44% | 0 |
| code_8192 | 8314 | draft-dflash | 3 | 3.37 | 12.73 | 13.19 | 1.03× | 68.60 | 11% | 0 |
| document_2048 | 2115 | none | 3 | 0.67 | 4.44 | 4.98 | 1.00× | 60.49 | — | 0 |
| document_2048 | 2115 | draft-eagle3 | 3 | 0.81 | 3.90 | 4.22 | 1.14× | 80.95 | 38% | 0 |
| document_2048 | 2115 | draft-dflash | 3 | 0.84 | 3.98 | 4.12 | 1.12× | 91.74 | 13% | 0 |
| document_512 | 578 | none | 3 | 0.22 | 3.37 | 3.90 | 1.00× | 62.75 | — | 0 |
| document_512 | 578 | draft-eagle3 | 3 | 0.25 | 2.59 | 2.91 | 1.30× | 79.76 | 32% | 0 |
| document_512 | 578 | draft-dflash | 3 | 0.30 | 2.81 | 2.91 | 1.20× | 76.18 | 10% | 0 |
| document_8192 | 8258 | none | 3 | 2.75 | 6.62 | 7.34 | 1.00× | 54.20 | — | 0 |
| document_8192 | 8258 | draft-eagle3 | 3 | 3.36 | 8.12 | 8.45 | 0.82× | 68.69 | 34% | 0 |
| document_8192 | 8258 | draft-dflash | 3 | 3.20 | 7.99 | 8.19 | 0.83× | 50.36 | 6% | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.

- chat_short, draft-eagle3: 0/3 exact outputs match baseline.
- chat_short, draft-dflash: 0/3 exact outputs match baseline.
- code_512, draft-eagle3: 0/3 exact outputs match baseline.
- code_512, draft-dflash: 0/3 exact outputs match baseline.
- code_8192, draft-eagle3: 0/3 exact outputs match baseline.
- code_8192, draft-dflash: 0/3 exact outputs match baseline.
- document_2048, draft-eagle3: 0/3 exact outputs match baseline.
- document_2048, draft-dflash: 0/3 exact outputs match baseline.
- document_512, draft-eagle3: 0/3 exact outputs match baseline.
- document_512, draft-dflash: 0/3 exact outputs match baseline.
- document_8192, draft-eagle3: 0/3 exact outputs match baseline.
- document_8192, draft-dflash: 0/3 exact outputs match baseline.

Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.

## Matched comparison findings

Geometric mean of the six per-case median throughput ratios: EAGLE-3 1.384×; DFlash 1.250×. EAGLE-3 increased throughput in all six cases; DFlash in five. First visible answer improved in five cases with EAGLE-3 and four with DFlash. Both delayed the visible answer for the 8K document case. These results include reasoning-token decoding and do not establish general model quality.

Validation: all 54 unique requests completed normally, all prompt-cache counts were zero, all 27 document answers matched the expected JSON, and all 18 code outputs matched the reviewed correct function AST. Three repeats characterize this small prompt suite, not all workloads.

Final settings: 32K context; target microbatch 128; Q8 target and draft KV caches; Q8 draft weights; EAGLE max draft 3, DFlash max draft 15. No tuning sweep was performed. A temporary 16GB Windows pagefile on E: resolved startup allocation failures (see host-memory-recovery.md). All target/draft layers and KV caches were GPU-resident according to startup logs. Earlier runs use different settings and are excluded.
