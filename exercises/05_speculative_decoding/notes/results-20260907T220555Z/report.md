# Speculative decoding benchmark

Complete run.

Same Qwen3-8B Q4_K_M target, 32768-token context, q8_0 KV cache, GPU layers all, Flash Attention on, thinking on, greedy sampling, and 1536-token output cap. Warm requests use cache_prompt=false. Client runs on GPU Station to isolate inference; absolute times are not directly comparable to the earlier Mac-to-Ollama baseline. Each repetition rotates method order and shuffles prompt order identically across methods. Exact loading mode and draft quantization are recorded in command files.

Values below are medians. Speedup is baseline time divided by method time: above 1 means faster. Null visible-answer times indicate no answer before the output limit.

| Prompt | User tokens | Method | n | TTFT s | Visible answer s | Total s | Visible speedup | Decode tok/s | Draft accepted | Truncated |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat_short | 12 | draft-eagle3 | 1 | 0.05 | 1.29 | 1.75 | —× | 111.56 | 41% | 0 |
| code_512 | 634 | draft-eagle3 | 1 | 0.20 | 5.11 | 5.66 | —× | 123.75 | 52% | 0 |
| code_8192 | 8314 | draft-eagle3 | 1 | 2.42 | 11.59 | 12.28 | —× | 96.86 | 47% | 0 |
| document_2048 | 2115 | draft-eagle3 | 1 | 0.59 | 3.84 | 4.08 | —× | 104.61 | 42% | 0 |
| document_512 | 578 | draft-eagle3 | 1 | 0.19 | 2.66 | 2.91 | —× | 98.99 | 35% | 0 |
| document_8192 | 8258 | draft-eagle3 | 1 | 3.41 | 5.98 | 6.27 | —× | 83.84 | 36% | 0 |

## Output checks

Exact full-output hashes are compared within the same repetition. Differences can arise from numerical or speculative sampling paths and must not automatically be interpreted as quality failures. Truncated runs are unsuitable for whole-answer latency comparisons.


Startup times and exact commands are in *-command.json. Per-request token counts, backend timing, and answers are in results.jsonl. Draft acceptance statistics are in server logs/metrics where exposed. Prompts are preserved in prompts.json.

N-gram configuration: match=12, min draft=4, max draft=16. EAGLE-3: RedHatAI-derived GGUF, max draft=3. DFlash: Z Lab-derived Q8 GGUF, max draft=15. See command files for exact quantization. These are initial configurations, not a search for the optimal draft length.
