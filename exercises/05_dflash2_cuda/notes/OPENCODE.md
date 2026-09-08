# OpenCode with GPUStation DFlash 2

Run `opencode-qwen` on the Mac. It selects `gpustation-dflash2/qwen3.8-27b-dflash2` for both the main and small model.

The shared authenticated endpoint is now `http://127.0.0.1:18080/v1`; the API key is already in `~/.config/opencode/opencode.json`. Qwen loads on the first request, switches safely after any active request finishes, and unloads after about two idle minutes. Run `opencode-nemotron` to use Nemotron instead.

Qwen retains the benchmarked UD-IQ2_S target and DFlash 2 Q4_K_M draft, all GPU layers, Q8 KV, 16,384 context, seven speculative tokens, and thinking disabled. OpenCode allows up to 4,096 output tokens.

The enabled service is `gpustation-router.service`; the old standalone `dflash2.service` is disabled. Windows, WSL/systemd, and Tailscale must be running.

See [the on-demand setup guide](../nemotron/README.md) for lifecycle, service commands, validation results, and limitations.
