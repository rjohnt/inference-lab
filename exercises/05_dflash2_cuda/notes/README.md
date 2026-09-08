# Qwen3.8-27B UD-IQ2_S versus DFlash 2

Status: COMPLETE. All 120 measured requests finished on GPUStation at 21:51 CDT on September 7, 2026. Report, PNG/SVG/PDF heatmaps, raw results, telemetry and validation are copied locally in `results-16k/`. All 60 paired outputs match exactly; zero retrieval failures or truncations. The 40 code AST differences were reviewed; see `results-16k/code-review.md`. The initial port-18081 startup failure was resolved by resuming once the port was free.

Remote directory: `/path/to/data/gpustation-benchmarks/dflash2`.

The target is pinned to `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`, file `Qwen3.8-27B-UD-IQ2_S.gguf`. The draft is pinned to `z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f8ce46e151f61c6499c99dee6079e1d610`, file `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`. `download_models.py` saves and verifies SHA256 hashes against pinned Hub metadata. All downloaded files and caches live on E: because C: has little free space.

## Benchmark design

- Six exact prompt texts from `specdec/results-20260908T010140Z/prompts.json`, retokenized for Qwen3.8.
- No speculation versus DFlash 2, seven speculative tokens per block.
- Ten repetitions per prompt and mode, 120 measured requests total.
- Thinking disabled, greedy sampling, seed 42, output cap 1536, uncached prompts.
- Alternate method order per repetition; shuffle prompts identically within paired repetitions.
- Same target, 16K context candidate, batch 512, microbatch 128, Q8 KV, Flash Attention, all target/draft layers on GPU. Context and memory fit must be verified with the probe before the measured run. Do not silently truncate the 8K cases to fit a 4K context.
- Fresh server per method and repetition, separate startup timing and an unmeasured warmup.
- Append and fsync every measured request. Restarting with the same `--out` skips completed requests and rejects changed configuration.
- The renderer requires all 120 unique requests. Reports include truncation, exact-output agreement, JSON retrieval correctness, and code AST differences for review. AST differences alone are not proof of functional failure.

## Persistent execution

tmux survives SSH/client disconnects. It does not survive a Windows/WSL shutdown. The runner's result journal supports resuming after a shutdown.

After downloads complete and files have been transferred, run a separate fit/smoke test first:

```bash
cd /path/to/data/gpustation-benchmarks/dflash2
python3 -u bench.py --probe --context 16384 --ubatch 128 --out probe-16k
```

Inspect probe logs for target and draft GPU placement, DFlash 2 selector metadata, prompt/output fit, memory, draft acceptance, and output quality before launching the measured run:

```bash
tmux new-session -d -s qwen38-benchmark 'cd /path/to/data/gpustation-benchmarks/dflash2 && python3 -u bench.py --context 16384 --ubatch 128 --repeats 10 --out results-16k > benchmark.log 2>&1'
```

Attach from WSL with `tmux attach -t qwen38-benchmark`; detach with Ctrl-B then D. Inspect progress with `cat /path/to/data/gpustation-benchmarks/dflash2/results-16k/status.json`.

The launched pipeline redirects to `pipeline.log`; to see live output from an attached shell use `tail -f /path/to/data/gpustation-benchmarks/dflash2/pipeline.log`. Downloads use `download.log`, and artifact rendering uses `render.log`.

Resume by rerunning the exact benchmark command with the same output directory. Do not launch a duplicate active session; the runner also holds an exclusive process lock.

On the Mac, after copying back the complete results:

```bash
python3 analyze.py results-16k
```

This generates the report and PNG/SVG/PDF heatmap. It compares the same target quantization with and without speculation; it does not isolate DFlash 1 versus DFlash 2 across different target models.

## Connection

```bash
ssh -o BatchMode=yes -o UseKeychain=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_example gpustation
```

`UseKeychain=yes` is required for the encrypted Mac private key. Do not ask the user to replace remote authorized keys when the server accepts the key but the local key is locked.
