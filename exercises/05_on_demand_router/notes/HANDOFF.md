# Current status — setup completed 2026-09-08

Resumed and completed all remaining verification. See README.md for current usage. Earlier sections below are historical and their remaining checklists are now complete.

- Overlap result existed and passed: Nemotron streaming completed with DONE, then Qwen answered 391; 41.54 seconds including queue/load. Remote journal confirmed ordering.
- Inspected live Qwen child: exactly one llama-server, full DFlash flags retained (draft-dflash, Q4_K_M draft, all draft GPU layers, n-max 7, Q8 draft KV). GPU usage 11,647 MiB.
- Switch back passed in 21.41 seconds; switch-back-check.json.
- Routed OpenCode read and shell assertion task passed; harness-router-retry.jsonl. Initial routed attempt used `python`, absent on Mac; explicit `python3` succeeded. No fixture edits.
- Verified both installed launchers match source, are executable, both provider URLs use 18080, and both timeouts are 600000. Keys preserved.
- Router enabled/active; standalone nemotron and dflash2 disabled/inactive. Local router.py/models.ini hashes matched deployed files.
- Idle: last response at 11:23:05 CDT, process unloaded at 11:25:06 (121 seconds); subsequent SSH showed no llama-server and 501 MiB baseline GPU usage. One HTTP status poll timed out during observation; cause not established, SSH and endpoint subsequently worked.
- Wake: both models listed unloaded before request; Nemotron returned 391 in 16.41 seconds, then listed loaded. idle-wake-check.json.
- Created README.md and replaced stale ../dflash2/OPENCODE.md with shared-router instructions.

Final operation: `opencode-nemotron` or `opencode-qwen`. Shared endpoint http://127.0.0.1:18080/v1. Windows/WSL/Tailscale must be up. One request at a time; unbounded, non-FIFO queue can exceed client timeout. Model status includes initializing processes. Router remains enabled; final wake left Nemotron resident until automatic idle unload.

---

# Resume GPUStation Nemotron + on-demand OpenCode setup

User is restarting Codex to refresh permissions. Resume from here; do not repeat downloads or request authorization again for the already-authorized setup.

## User requests

1. Set up Nemotron on GPU Station.
2. Selected **Nemotron Nano 9B v2** after hardware inspection.
3. Expose an endpoint usable with the **OpenCode harness** on the Mac.
4. Latest requirement: expose both Nemotron and existing Qwen/DFlash models on demand; load only when requests arrive. Implemented native llama-server router, pending verification and client migration.

## Connection and hardware

Mac workspace: `/path/to/local-home/benchmarks/gpustation/nemotron`.
SSH: `ssh -o BatchMode=yes -o UseKeychain=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -i ~/.ssh/id_ed25519_example gpustation`
Host alias -> `<linux-user>@127.0.0.1`, WSL on Windows `<windows-host>`.
Encrypted key needs `UseKeychain=yes`; do not replace keys.
RTX 4070 SUPER, 12282 MiB VRAM; WSL RAM 7.7 GiB + 2 GiB swap.
GPU command: `/usr/lib/wsl/lib/nvidia-smi` (not on normal PATH).
Use E: for large files; `/path/to/data` has ~880 GiB free. Remote `rg` unavailable.

## Completed and validated

Downloaded and SHA256 verified:
`/path/to/data/gpustation-models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf`
6,525,629,280 bytes; repo `bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF`.
Revision `b2daccafb3d123ea94653720a0db1f5db9c698ff`.
SHA256 `be84528231457f766a35612386e2638d9af10beb10a9d8d23f0cc73c20818da7`.
Manifest copied locally as `manifest.json`. Download script/log on remote in same model directory.

Standalone Nemotron service worked on `http://127.0.0.1:18083/v1`, alias `nemotron-nano-9b-v2`, 16384 context, Q8 KV, all GPU layers, same existing llama-server CUDA runtime. About 6909 MiB total GPU usage. ~17-second initial model load. Tiny generation ~54 tokens/s, not a rigorous benchmark.
`check_endpoint.py` passed authentication (401 without key), chat, structured tool-call arguments, and tool-result round trip. Results in `endpoint-check.json`.
OpenCode 1.18.29 recognized provider and successfully read, edited, and ran assertions in `harness-smoke/calculator.py`. Result in `harness-smoke-retry.jsonl`. Initial trial guessed `/path/to/calculator.py`; OpenCode rejected that external path. Retry with explicit real path passed all steps. Fixture now contains corrected `return a + b`.
Nemotron defaults to thinking; `/no_think` in system/user message disables thinking. Smoke tests used `/no_think`. NVIDIA recommends greedy when no-thinking, temp .6/top_p .95 when thinking.

## Previous attempt: native router (SUPERSEDED by update below)

The installed `/usr/local/lib/ollama/llama-server` is version `0.3.0-dev`, commit `0f3a71be1`; supports native router, presets, models-max, sleep-idle-seconds and DFlash 2.

Files created locally and copied to `/path/to/data/gpustation-models/nemotron/`:
- `models.ini`: two presets, no load on startup; 120-second idle sleep; all GPU layers, 16K context, Q8 KV, DIO. Qwen retains original DFlash settings.
- `gpustation-router.service`: one always-running lightweight router on **127.0.0.1:18080**, `--models-max 1`, `--timeout 600`, same existing API key file.

Last remote command succeeded:
```
cp /path/to/data/gpustation-models/nemotron/gpustation-router.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user disable nemotron.service dflash2.service
systemctl --user enable --now gpustation-router.service
```
Service journal only showed "Started" at that moment; MUST inspect logs and health for argument/config errors. Router conflicts with old standalone services, so should have stopped Nemotron. Old service definitions retained, both disabled at boot. Do not run old services concurrently with router.

Native routing: same OpenAI-compatible base URL `http://127.0.0.1:18080/v1`, JSON `model` chooses `nemotron-nano-9b-v2` or `qwen3.8-27b-dflash2`.
No need for a custom proxy if native behavior validates.

## Previous client state (SUPERSEDED by update below)

`/path/to/local-home/.config/opencode/opencode.json`:
- provider `gpustation-nemotron` added, model `nemotron-nano-9b-v2`, 16384 context, 4096 output, tool_call true. **Still points at port 18083.**
- provider `gpustation-dflash2` existing, model `qwen3.8-27b-dflash2`. **Still points at port 18082.**
- default `model` and `small_model` both now `gpustation-nemotron/nemotron-nano-9b-v2`.
- Both provider API keys are same existing DFlash key. Preserve; do not print secrets.
- Other unrelated providers untouched.
- Backup before changes: `opencode.json.backup-nemotron-20260908-100252`.

Update both baseURLs to `http://127.0.0.1:18080/v1` after confirming router works. Consider provider timeout 600000 if SDK supports it, to tolerate cold starts.
Local `opencode-nemotron` source was revised to only set small_model and exec OpenCode; no SSH/manual startup needed anymore. **Installed `~/.local/bin/opencode-nemotron` is still the OLD version that starts standalone service! Must replace it before user uses it.**
New `opencode-qwen` source sets Qwen small_model and model; not installed yet. Install both from this directory into `~/.local/bin` with mode 755.
Small-model selection matters: fixed Nemotron small_model when using Qwen could otherwise trigger needless switching for titles/other background tasks. Launchers set it to match selected main model.

## Original remaining work (see current checklist below)

1. Inspect router service log and `/health`, authenticated `/v1/models`; verify both listed **unloaded** before any inference and near-zero GPU model usage.
2. Fix any preset argument incompatibilities without changing original Qwen inference settings.
3. Migrate OpenCode baseURLs and install revised launchers.
4. Test cold Nemotron request, switch to Qwen with a request, switch back; verify only one model resident and responses succeed. Verify DFlash args are retained in child process/model status. Check behavior with overlapping requests: do not assume router safely waits rather than interrupting an active other-model request; report any relevant limitation.
5. Test streaming/tool use through router (existing endpoint test reads config, so will target migrated URL). OpenCode smoke passed direct endpoint; validate at least a routed OpenCode tool task too.
6. Wait/check 120-second idle sleep actually releases GPU VRAM; next request wakes model. Use short polling/tool waits and communicate every ~60 seconds.
7. Write concise README with final endpoint, model IDs, launch commands, startup behavior (Windows/WSL must be up), idle policy/cold-start latency, and test results. Update old `../dflash2/OPENCODE.md` to point at router so it no longer advertises stale endpoint/service instructions.
8. Final answer should give launch commands and shared endpoint, confirm automatic model loading/switching/idle unload only after verified. User prefers concise actionable replies.

## References checked

- https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
- https://huggingface.co/bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF
- https://opencode.ai/docs/providers/
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md (sections Using multiple models, Model presets, Sleeping on Idle)
- https://github.com/ggml-org/llama.cpp/blob/master/docs/autoparser.md supports NVIDIA-Nemotron-Nano-v2 TOOLCALL parser.

Native presets: `version = 1`, `[*]` common settings, `[model-id]` per model. Keys are command flags without dashes. `load-on-startup=false`. Router CLI arguments override presets. `--models-max 1` limits loaded children; `--sleep-idle-seconds 120` unloads model/KV memory after idle, auto-wakes on request. `/health`, `/props`, `/models`, `/metrics` don't reset idle or wake sleeping models (avoid unverified assumptions for initially unloaded routing GETs). API authentication key is `/path/to/remote-home/.config/dflash2/api-key` on GPUStation; stored in existing local provider config too.

No subagents used; user did not request delegation. No applicable AGENTS.md was found in checked local ancestor/benchmark paths or remote benchmark directory.


# Latest handoff — 2026-09-08, approximately 11:21 CDT

User asked to update handoff and restart Codex. Resume from THIS section; it supersedes the native-router plan and stale-client notes above. User changed permissions to full access / never approvals before restart. Respect current session permissions on resume. Setup remains authorized; do not ask again or re-download models.

## Root cause and replacement deployed

Native router could not run: journal repeatedly reported `failed to initialize router models: subprocess is not enabled on this build`. Stopped the restart loop. Existing llama-server supports inference but its native subprocess router was compiled out.

Created `router.py` in this directory (Python standard library only), copied to `/path/to/data/gpustation-models/nemotron/router.py`. Replaced local and deployed `gpustation-router.service` ExecStart with:
`/usr/bin/python3 /path/to/data/gpustation-models/nemotron/router.py`
Daemon reloaded, service restarted, verified active. Same environment/CUDA libraries, conflicts with old standalone services, enabled at boot. Old `nemotron` and `dflash2` services remain disabled and stopped.

Replacement behavior:
- Public endpoint `http://127.0.0.1:18080/v1`, same existing API key (never print it).
- Child llama-server binds only `127.0.0.1:18081` with same API key.
- Reads `models.ini` for both models, preserving Qwen inference/DFlash args. Skips native `load-on-startup` and `sleep-idle-seconds` arguments because Python owns lifecycle.
- Authenticated `/v1/models` and `/models` list both models and `loaded`/`unloaded` status; unauthenticated `/health` is lightweight.
- Proxies `/v1/chat/completions` and `/v1/completions`, including streaming.
- Single lock covers loading AND entire response; other inference requests wait, so switching cannot kill an active response. No parallel inference. Queue is not explicitly FIFO and has no bound; long queues may exceed client timeout.
- Model startup deadline 540 seconds, backend HTTP timeout 600 seconds.
- Idle monitor stops child process 120–122 seconds after last completed request; no model loaded on router startup. This is process termination, not native sleep.
- On client disconnect or backend error, stops child before releasing lock.
- `/v1/models` currently reports process presence, so `loaded` may appear while model is still initializing; health only proves router is up.
- Python syntax compiled locally. Full lifecycle tests still in progress.

## Client migration DONE

Updated `/path/to/local-home/.config/opencode/opencode.json`:
- Both `gpustation-nemotron` and `gpustation-dflash2` baseURL now `http://127.0.0.1:18080/v1`.
- Both options timeout set to `600000` milliseconds (still need routed OpenCode run to validate this config in practice).
- Preserved API keys, model metadata, unrelated providers and defaults.
- Backup created next to config as `opencode.json.backup-router-<timestamp>`.
Installed revised local `opencode-nemotron` and new `opencode-qwen` into `~/.local/bin`, mode 755. They exec `~/.opencode/bin/opencode` and set small_model to match chosen model. No SSH/start-service commands in either installed launcher.

## Verified on replacement router

`python3 benchmarks/gpustation/nemotron/check_endpoint.py` PASSED:
- `/v1/models` showed BOTH UNLOADED before initial inference.
- Missing API key returned 401.
- Cold Nemotron chat answered 391; request took 21.50 seconds including load.
- Structured get_weather tool call and tool-result round trip passed.
- Results overwrite local `endpoint-check.json` (now routed results).
- Nemotron loaded GPU memory ~6907 MiB; baseline before replacement was ~501 MiB.

## Test STILL RUNNING at handoff

Created and launched `check_router.py` in this directory. It starts a long streaming Nemotron response (600 output tokens), then sends Qwen request after 1 second in another thread. Asserts Nemotron receives `[DONE]`, Qwen returns 391, and writes `router-check.json` with timing/content/status after both finish.

Local process at last check: PID **13928**, command `python3 benchmarks/gpustation/nemotron/check_router.py`. Old exec session **82661** may not survive restart; check process and output file before launching another request. `router-check.json` did NOT yet exist at last check. Script has 600-second request timeout; if it fails, no result file is written and error was only in prior exec output.

Remote last observation:
- Router active.
- Nemotron finished 600 tokens at 11:20:06 (~9.64 seconds generation+prompt).
- Router THEN unloaded Nemotron at 11:20:06, THEN launched Qwen child PID 210688.
- Qwen target initialization reached 11:20:32, then started loading DFlash draft.
- GPU memory ~11195 MiB at last check.
This supports correct ordering but does NOT yet establish full Qwen success / completed overlap test. Inspect journal and process, especially if Qwen draft load stalls or OOMs. Do not claim overlap test passed without evidence.

## Remaining checklist, in order

1. Check running overlap test, `router-check.json`, router journal, child process and GPU memory. Confirm Qwen completes and streaming Nemotron received DONE. Diagnose any failure; source settings preserve prior Qwen inference settings (old standalone also had `--log-verbosity 4`, omitted in preset but not an inference setting).
2. Verify running Qwen child includes all DFlash draft flags and only one model process exists.
3. Switch back to Nemotron and confirm successful response. Use helpers in `check_router.py` (importing it does NOT execute tests; main guard present). Do NOT import `check_endpoint.py` casually, since it runs its tests at import.
4. Run a real routed OpenCode tool task via installed launcher. Existing fixture `harness-smoke/calculator.py` has correct `add(a,b): return a+b`. Ask it to read absolute fixture path and use shell to assert add(2,3)==5 and add(-2,2)==0, or another bounded task. Prior direct-endpoint test passed; routed harness remains untested. Save JSON output in a new `harness-router.jsonl`. `opencode run --format json --dir <fixture-dir> ...` available. Avoid accidentally sharing/publishing session.
5. Observe 120-second idle unload with no intervening inference, record models status + near-baseline GPU memory, then wake with request. Poll/wait in short intervals and communicate approximately every 60 seconds.
6. Create concise `README.md` here with final endpoint, model IDs, installed launchers, automatic startup/loading/serialization/idle behavior, measured cold start and tested limitations. Windows + WSL/systemd must be up; no Windows auto-start changes were made. Native router is unavailable in installed build, so custom stdlib proxy is used.
7. Update `../dflash2/OPENCODE.md` — still stale port 18082 and standalone service instructions! Point to shared router/launchers and new README.
8. Update this handoff with completed results and final concise answer to user. No full completion claim until checks above pass.

No agents delegated. No AGENTS.md found in relevant ancestors. No extra models downloaded or dependencies installed in resumed turn.
