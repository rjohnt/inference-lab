# MacBook Qwen3.8 benchmark

Status: COMPLETE. All 120 Mac responses completed on September 8, 2026. The updated four-column report and PNG/SVG/PDF heatmaps are in `comparison/` and `/path/to/local-home/Downloads/Qwen3.8-DFlash2-Benchmark/`. All 60 Mac baseline/DFlash pairs match exactly. Across both machines there are zero retrieval failures, Python syntax failures, or truncations. Original GPU-only Downloads artifacts are preserved in `gpu-only/`.

All GPU benchmark settings are matched: 16,384 context, exact same six prompts, ten repetitions for baseline and DFlash 2, thinking off, greedy decoding, seed 42, 1,536 output cap, batch 512, microbatch 128, Q8 KV, one slot, all layers on GPU, no prompt reuse, seven-token speculative blocks. Target is the existing Mac Qwen3.8-27B Q4_K_M file. Draft is the exact SHA256-verified GPUStation DFlash 2 Q4_K_M file. GPUStation used an IQ2_S target, so cross-machine results are configuration comparisons.

Ollama’s bundled llama-server and the installed Homebrew runner failed the DFlash 2 tensor-format probe. Both Mac modes therefore use an isolated Metal build of the GPUStation source revision, `ggml-org/llama.cpp@0f3a71be1`. Sources and build log are retained here. This tests the same Mac model weights that were imported into Ollama, with direct llama-server API controls to match the GPU benchmark’s cache policy. It does not time the Ollama HTTP scheduling layer.

A 12-response probe passed before the full measurement. Fresh server per mode and repetition; unmeasured warmup; startup excluded. Each result is appended and fsynced, and duplicate keys/configuration changes are rejected on resume. The Mac is connected to AC power and caffeinate prevents idle sleep while the job runs.

Run `python3 compare.py` after completion to generate the four-column report and PNG/SVG/PDF heatmap in `comparison/`. Raw results, environment, checksums, command lines, GPU offload logs, and memory telemetry remain in `results-16k/`.
