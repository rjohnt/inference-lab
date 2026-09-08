# GPUStation on-demand models

Launch from the Mac:

```sh
opencode-nemotron
opencode-qwen
```

Both use the authenticated OpenAI-compatible endpoint `http://127.0.0.1:18080/v1`. The existing API key is already configured in `~/.config/opencode/opencode.json`. The launchers select matching main and small models to avoid background requests switching models unnecessarily. The default is Nemotron.

| Launcher | API model ID | Model |
| --- | --- | --- |
| `opencode-nemotron` | `nemotron-nano-9b-v2` | Nemotron Nano 9B v2 Q4_K_M |
| `opencode-qwen` | `qwen3.8-27b-dflash2` | Qwen3.8 27B UD-IQ2_S with DFlash 2 Q4_K_M draft |

Both have 16,384 context and up to 4,096 output tokens in OpenCode. Nemotron supports `/no_think` in prompts; Qwen reasoning is disabled. Qwen retains all-GPU target/draft loading, Q8 KV, and seven speculative tokens.

## Lifecycle

`gpustation-router.service` starts with WSL/systemd. It initially loads no model. An inference request loads the selected model; switching unloads the previous model first. Requests are serialized through the entire response, including streaming. After 120–122 seconds without inference, the child process is stopped to release model GPU memory. A subsequent request loads it again.

Windows, WSL, and Tailscale must be running. No Windows auto-start changes were made. The old `nemotron.service` and `dflash2.service` are disabled; use the router service. Other GPU applications, including separately loaded Ollama models, still compete for VRAM.

The installed llama-server build has native subprocess routing disabled, so `router.py` supplies the lifecycle and proxy using Python's standard library. `models.ini` holds model settings; the backend binds only to `127.0.0.1:18081`.

On GPUStation:

```sh
systemctl --user status gpustation-router
systemctl --user restart gpustation-router
systemctl --user stop gpustation-router
journalctl --user -u gpustation-router -n 40 --no-pager
```

## Validation and limits

- Authenticated model listing and rejection without a key passed.
- Routed Nemotron chat, structured tool calls, and tool-result round trip passed (`endpoint-check.json`). Initial cold request took 21.50 seconds.
- Concurrent Nemotron streaming and Qwen request passed: Nemotron received the final stream marker before switching; Qwen returned the correct answer. Qwen took 41.54 seconds including queue time, roughly 31 seconds to load (`router-check.json`).
- Routed OpenCode read the fixture and ran both calculator assertions successfully (`harness-router-retry.jsonl`). The initial attempt used unavailable `python`; specifying `python3` passed.
- Wake after idle passed in 16.41 seconds with the correct answer (`idle-wake-check.json`).
- Idle unload occurred 121 seconds after the last response; no llama-server process remained and GPU usage returned to 501 MiB. One HTTP status poll timed out during observation; subsequent SSH confirmed the service remained active.
- One llama-server process was present with all Qwen DFlash arguments, using about 11,647 MiB GPU memory. Switching back to Nemotron passed in 21.41 seconds (`switch-back-check.json`). These small requests are functional checks, not performance benchmarks.

Only one inference runs at a time. The queue is unbounded and not guaranteed FIFO; long queues can exceed the configured 600-second client timeout. Model startup has a 540-second deadline. `/health` checks the router only; `/v1/models` reports process presence, so `loaded` can include initialization. Only chat completions and text completions are proxied; this is not a full OpenAI API implementation.
