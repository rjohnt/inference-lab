# Qwen3.8 27B: two independent DFlash 2 benchmarks

Each benchmark compares No Speculation against DFlash 2 on one fixed device and target model. Target quantizations differ between benchmarks; these results do not establish relative hardware performance.

Both use 16,384 context, the same six prompt texts, 10 repetitions per case/method, thinking off, temperature 0, seed 42, maximum output 1,536 tokens, Q8 KV, batch 512, microbatch 128, one slot, and uncached prompts. DFlash 2 uses seven draft tokens and the same Q4_K_M draft file. Runner revision: 0f3a71be1.

Fresh server per method/repetition, one unmeasured warmup, alternating method order and seeded prompt shuffles. Mac weights are loaded directly by llama.cpp, bypassing the Ollama HTTP scheduler. Its OpenCode configuration remains 64K; this benchmark uses 16K.

## MacBook M5 Pro

Q4_K_M target · 64 GB unified memory · macOS / Metal

120 measured responses. Speedup is relative only to this benchmark’s No Speculation baseline.

| Case | No Speculation (tok/s) | DFlash 2 (tok/s) | Decode speedup | Baseline request (s) | DFlash request (s) | Exact output agreement |
|---|---:|---:|---:|---:|---:|---:|
| chat_short | 13.53 | 13.43 | 0.99× | 6.057 | 6.161 | 10/10 |
| document_512 | 13.51 | 30.42 | 2.25× | 5.780 | 3.964 | 10/10 |
| document_2048 | 13.44 | 30.18 | 2.25× | 10.371 | 8.903 | 10/10 |
| document_8192 | 13.04 | 28.36 | 2.17× | 30.694 | 30.696 | 10/10 |
| code_512 | 13.50 | 30.83 | 2.28× | 9.824 | 5.916 | 10/10 |
| code_8192 | 13.00 | 31.13 | 2.39× | 36.348 | 33.700 | 10/10 |

## RTX 4070 Super

IQ2_S target · 12 GB VRAM · Linux / CUDA

120 measured responses. Speedup is relative only to this benchmark’s No Speculation baseline.

| Case | No Speculation (tok/s) | DFlash 2 (tok/s) | Decode speedup | Baseline request (s) | DFlash request (s) | Exact output agreement |
|---|---:|---:|---:|---:|---:|---:|
| chat_short | 42.90 | 55.71 | 1.30× | 2.223 | 1.893 | 10/10 |
| document_512 | 42.51 | 122.23 | 2.88× | 1.968 | 1.491 | 10/10 |
| document_2048 | 42.12 | 121.89 | 2.89× | 3.513 | 3.137 | 10/10 |
| document_8192 | 40.10 | 110.47 | 2.75× | 9.084 | 9.443 | 10/10 |
| code_512 | 42.64 | 132.54 | 3.11× | 3.240 | 1.968 | 10/10 |
| code_8192 | 40.55 | 134.94 | 3.33× | 10.939 | 10.221 | 10/10 |

240 responses: 0 retrieval failures; 0 Python syntax failures; 0 truncated.
Python syntax checks do not establish functional correctness. Decode throughput excludes prefill and startup; request time includes prefill and decoding, excluding startup and warmup.

Raw Mac run: ../results-16k/; GPU run: ../../../gpustation/dflash2/results-16k/. Configurations, checksums, outputs, logs and timings are retained.
