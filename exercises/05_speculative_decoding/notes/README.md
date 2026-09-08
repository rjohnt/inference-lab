# GPU Station speculative decoding benchmark

## Status

**Correction: this run used DFlash 1, not DFlash 2. The upstream Qwen3-8B-DFlash-b16 draft is documented for thinking disabled, but this run enabled thinking for all methods. Results describe that tested configuration and should not be treated as a comparison of optimally configured EAGLE and DFlash. This mismatch may contribute to low draft acceptance; its effect has not been isolated.**

Source: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16


Latest completed comparison: `results-20260907T225751Z`, 180 requests, N=10 per cell. Its report, uncertainty intervals, validation, host telemetry and heatmap are saved locally. EAGLE-3 geometric-mean throughput speedup: 1.332×; DFlash: 1.203×. The N=3 run below is historical.

The matched no-speculation / EAGLE-3 / DFlash comparison completed in `results-20260907T224213Z`: three repetitions, six prompt cases per method, 54 measured requests, 32768-token context, 128-token target microbatch and Q8 target/draft KV caches. All 54 requests completed with uncached prompts and no truncation. All retrieval answers were correct and every code answer matched the reviewed correct function. Previous incomplete and single-pass runs remain separate.

Across six equally weighted prompt cases, the geometric mean of per-case median throughput ratios was 1.384× for EAGLE-3 and 1.250× for DFlash. EAGLE-3 improved decode throughput in all six cases; DFlash improved five and regressed on the 8K document case. DFlash narrowly led the short code task and clearly led the 2K document task. These are results for the tested backend, Q8 drafts and draft lengths, not a general ranking of algorithms.

The generated heatmap, full report, raw measurements, and validation results are in the completed run directory.

DFlash successfully loaded and generated at 32K after adding temporary Windows paging capacity on E:. The preceding 32K and 16K CUDA allocation failures were not sufficient evidence of a physical VRAM capacity limit. See `host-memory-recovery.md` for the exact host change and its lifetime. Target layers, draft layers, and KV caches are on the GPU; host staging/model buffers remain.

DFlash draft SHA256 verified against pinned repository metadata: `5be4f6b1bfd5c2c1aa753d4c03e30700114654fefbbf29f02257ef37adb00bf0`. The 1,120,250,336-byte file is on E:, linked from the benchmark models directory to avoid filling C:.

EAGLE-3 uses a Q8_0 quantization of the pinned RedHatAI-derived community GGUF. Original F16 SHA256: `d6cf1f3cf29e9cd72c02fb11f989f5192f2b24e142741fdc2de8cd590140f2f2`.

## Prompt matrix

| Case | Context size | Task |
|---|---|---|
| chat_short | One sentence | Explain a diffusion model in three short sentences |
| document_512 | 512-token inventory plus instructions | Extract a marked service record as JSON |
| document_2048 | 2,048-token inventory plus instructions | Same extraction task |
| document_8192 | 8,192-token inventory plus instructions | Same extraction task |
| code_512 | 512-token code excerpt plus target function | Fix empty-input handling; return updated function |
| code_8192 | 8,192-token code excerpt plus target function | Same code edit with long context |

Document and code payloads are synthesized deterministically. The target server tokenizer controls context length, and actual user token counts are saved. Identical prompts are used across all methods and saved in each result directory's `prompts.json`.

## Methods and controls

- DFlash: `--spec-type draft-dflash`, Q8_0 draft, maximum draft length 15. Uses the official Z Lab Qwen3-8B-b16 draft converted by AtomicChat; pinned URL is in runner metadata. Runtime-validated at 32K.
- No speculation: `--spec-type none`.
- N-gram: `--spec-type ngram-mod`, match 12, draft 4–16 tokens.
- EAGLE-3: `--spec-type draft-eagle3`, Q8_0 draft, maximum draft length 3.
- Same existing Qwen3-8B target weights; context 32768; one slot; Flash Attention enabled; q8_0 K/V cache; all target and draft layers requested on GPU; batch 512; final comparison microbatch 128; direct-I/O loading.
- Thinking enabled; greedy sampling; seed 42; maximum output 1536 tokens. Runs hitting the limit are flagged; a missing visible answer is not counted as a fast answer.
- Three repetitions; method order rotates by repetition; prompt order shuffled identically within each repetition.
- Fresh server per method/repetition, unmeasured warmup, then six measured requests. `cache_prompt=false` avoids prefix-cache advantages. N-gram state may still accumulate within a method block.
- Process startup-to-health measured separately; this is not a controlled disk-cold test.
- Client timing is local to GPU Station, excluding Tailscale. Compare methods within this suite; do not directly attribute differences from the earlier Mac/Ollama baseline to speculation.

## Reproduce

From the Mac, copy the runner and report generator:

```sh
scp /path/to/local-home/benchmarks/gpustation/specdec/bench.py /path/to/local-home/benchmarks/gpustation/specdec/report.py gpustation:benchmarks/specdec/
```

On GPU Station, first check no previous benchmark process is running and port 18081 is free. The runner uses that localhost-only port and unloads only `qwen3:8b-32k` from Ollama to avoid competing GPU allocations. Avoid other inference requests during the run.

```sh
cd ~/benchmarks/specdec
python3 -u bench.py --repeats 3 --ubatch 128 --modes draft-dflash none draft-eagle3
python3 report.py results-TIMESTAMP
```

For an initial runtime check, use `--repeats 1`. Default runner includes n-gram (72 requests); the command above selects the requested three methods (54 requests). Each run writes a new timestamped directory; original baseline files are untouched. Review logs for GPU offload and draft acceptance. The successful direct-I/O validation confirmed compatibility; retain failure checks when changing configurations.

## Outputs

- `results.jsonl`: TTFT, first visible answer, total time, backend timings, token usage, finish reason, output hash, answer.
- `*-command.json`: exact flags and startup duration.
- `*.log` and `*-metrics.txt`: server diagnostics and speculative statistics where exposed.
- `metadata.json`: backend version, GPU, environment, source URL and method.
- `report.md` / `summary.json`: median timings, visible-answer speedup, generation rate, output-limit flags, exact-output comparisons.
- `COMPLETE`: written only after every requested run succeeds.

A hash mismatch prompts review rather than proving a quality regression. These initial prompts measure latency, not general model quality; n-gram speculation may benefit more from larger repeated code output than this compact edit task.

## DFlash setup and context policy

Default context remains 32768. Preserve that unless a clean, healthy system cannot fit the draft. If context must be reduced, use `--context 16384` for **all methods in the comparison**, not only DFlash. The longest prompt plus output allowance fits at 16K. `--ubatch 128` is also available to reduce temporary compute buffers uniformly before reducing context.

DFlash artifact:
`https://huggingface.co/AtomicChat/Qwen3-8B-DFlash-GGUF/resolve/788b1a553f50979b99fecf6abe7a4c3fd88a8d89/Qwen3-8B-DFlash.Q8_0.gguf`

The draft is stored at `/path/to/data/gpustation-benchmark-models/dflash-q8.gguf` and symlinked to `~/benchmarks/specdec/models/dflash-q8.gguf`. `verify_dflash.py` validates its SHA256 against the pinned tree metadata. Context reduction did not resolve the initial allocation failures; adding Windows commit capacity allowed full 32K to run. See `host-memory-recovery.md`.

## Local analysis and image

```sh
python3 report.py results-20260907T224213Z
python3 check_results.py results-20260907T224213Z
python3 plot_comparison.py results-20260907T224213Z
```

The chart is rendered locally on the Mac using Matplotlib. Higher throughput uses darker cells on one shared color scale. Each cell shows the median tokens/second and its ratio to the baseline median for the same prompt. PNG, SVG and PDF are exported. The renderer refuses incomplete runs or cells without three measurements.
