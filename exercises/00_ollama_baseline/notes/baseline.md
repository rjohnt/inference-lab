# GPU Station unchanged baseline

Measured from the Mac over Tailscale using the OpenAI-compatible streaming endpoint. Qwen3 8B, 32K context, default thinking enabled, temperature 0.2, maximum output 2048 tokens. No persistent configuration changes.

TTFT counts the first nonempty reasoning or answer delta. First answer counts the first nonempty content delta. Role-only chunks are excluded. One cold sample; warm values are medians of three repetitions.

| Condition / prompt | TTFT | First answer | Total |
|---|---:|---:|---:|
| cold: ping | 13.35 s | 16.25 s | 18.44 s |
| warm: ping | 0.29 s | 2.18 s | 2.41 s |
| warm: what's a diffusion model? | 0.48 s | 9.32 s | 18.90 s |
| warm: what model are you | 3.49 s | 4.72 s | 5.28 s |

Warm TTFT range: 0.10–3.85 seconds.

These are standalone prompts, with no OpenCode system prompt, tools, or conversation history. They do not reproduce the full Build workload. Warm requests may reuse prompt cache. The initial cold sample explicitly unloaded only this model first. All requests should end with finish_reason=stop; raw usage counts and timing are in baseline-results.jsonl.

The original OpenCode logs showed roughly 10.7K input tokens, 63 tokens/second decoding, and 32.11 / 11.27 / 8.35 seconds server request duration. The UI durations supplied by the user were 31.6 / 11.0 / 9.9 seconds, with 7.8 / 8.8 / 7.6 seconds thinking. Neither UI total duration nor thinking duration is a direct TTFT measurement.

Re-run baseline.py only after archiving the current results: it overwrites baseline files. Use the same prompts, sampling parameters, client path, and cold/warm procedure when comparing improvements.
