# Speculative decoding benchmark

**Correction: this run used DFlash 1, not DFlash 2. The upstream Qwen3-8B-DFlash-b16 draft is documented for thinking disabled, but this run enabled thinking for all methods. Results describe that tested configuration and should not be treated as a comparison of optimally configured EAGLE and DFlash. This mismatch may contribute to low draft acceptance; its effect has not been isolated.**

Source: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16

Complete run.

Same Qwen3-8B Q4_K_M target, 32768-token context, q8_0 target KV cache (draft cache settings in command files), GPU layers all, Flash Attention on, thinking on, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | none | 10 | 0.06 | 1.78 | 2.50 | 1.00× | 79.53 | — | 0 |
| chat_short | 12 | draft-eagle3 | 10 | 0.05 | 1.30 | 1.77 | 1.36× | 110.58 | 41% | 0 |
| chat_short | 12 | draft-dflash | 10 | 0.09 | 1.91 | 2.49 | 0.93× | 85.51 | 8% | 0 |
| code_512 | 634 | none | 10 | 0.21 | 10.17 | 11.47 | 1.00× | 77.36 | — | 0 |
| code_512 | 634 | draft-eagle3 | 10 | 0.25 | 6.26 | 6.81 | 1.63× | 121.41 | 51% | 0 |
| code_512 | 634 | draft-dflash | 10 | 0.25 | 7.14 | 7.34 | 1.42× | 118.97 | 15% | 0 |
| code_8192 | 8314 | none | 10 | 2.42 | 10.82 | 12.32 | 1.00× | 67.26 | — | 0 |
| code_8192 | 8314 | draft-eagle3 | 10 | 2.94 | 9.53 | 10.22 | 1.14× | 91.99 | 44% | 0 |
| code_8192 | 8314 | draft-dflash | 10 | 2.85 | 10.64 | 11.04 | 1.02× | 81.75 | 11% | 0 |
| document_2048 | 2115 | none | 10 | 0.58 | 3.59 | 4.03 | 1.00× | 75.60 | — | 0 |
| document_2048 | 2115 | draft-eagle3 | 10 | 0.70 | 3.26 | 3.52 | 1.10× | 97.80 | 38% | 0 |
| document_2048 | 2115 | draft-dflash | 10 | 0.72 | 3.42 | 3.54 | 1.05× | 107.00 | 13% | 0 |
| document_512 | 578 | none | 10 | 0.19 | 2.71 | 3.14 | 1.00× | 78.36 | — | 0 |
| document_512 | 578 | draft-eagle3 | 10 | 0.22 | 2.22 | 2.49 | 1.22× | 93.94 | 32% | 0 |
| document_512 | 578 | draft-dflash | 10 | 0.25 | 2.34 | 2.43 | 1.16× | 91.41 | 10% | 0 |
| document_8192 | 8258 | none | 10 | 2.40 | 5.51 | 6.09 | 1.00× | 67.38 | — | 0 |
| document_8192 | 8258 | draft-eagle3 | 10 | 2.90 | 6.93 | 7.20 | 0.79× | 81.44 | 34% | 0 |
| document_8192 | 8258 | draft-dflash | 10 | 2.82 | 6.73 | 6.89 | 0.82× | 61.45 | 6% | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.

- chat_short, draft-eagle3: 0/10 exact outputs match baseline.
- chat_short, draft-dflash: 0/10 exact outputs match baseline.
- code_512, draft-eagle3: 0/10 exact outputs match baseline.
- code_512, draft-dflash: 0/10 exact outputs match baseline.
- code_8192, draft-eagle3: 0/10 exact outputs match baseline.
- code_8192, draft-dflash: 0/10 exact outputs match baseline.
- document_2048, draft-eagle3: 0/10 exact outputs match baseline.
- document_2048, draft-dflash: 0/10 exact outputs match baseline.
- document_512, draft-eagle3: 0/10 exact outputs match baseline.
- document_512, draft-dflash: 0/10 exact outputs match baseline.
- document_8192, draft-eagle3: 0/10 exact outputs match baseline.
- document_8192, draft-dflash: 0/10 exact outputs match baseline.

Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.

## Higher-N findings and host observations

All 180 requests completed: 10 repetitions per task and method at 32K context. Validation passed for normal completion, uncached prompts, retrieval answers, and code ASTs. Geometric mean of the six per-case median throughput ratios: EAGLE-3 1.332×; DFlash 1.203×. EAGLE led five tasks; DFlash led the 2K document task. DFlash was 0.912× baseline on the 8K document task. See uncertainty.md for paired bootstrap intervals and observed ranges.

GPU telemetry sampled every 2 seconds: 760 samples, 46–79°C, 592–9666MiB total device memory use including startup/idle intervals. Clock-event flags showed software power limiting in 376 samples, idle in 201, no active limit in 183; no thermal-limiting flag was observed. WSL pswpin and pswpout counters did not increase. Major faults did increase, which can include file-backed model loading and do not establish Windows pagefile activity. These measurements do not prove that Windows never evicted GPU allocations or paged host memory.

Absolute throughput was higher than the earlier N=3 run; use the matched comparisons within this run. No causal attribution for that between-run difference has been established. Increased N characterizes repeatability of the same six prompts, not workload diversity. Profiling/installation occurred after benchmark completion.
