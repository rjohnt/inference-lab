import json
import statistics
from pathlib import Path

p = Path(__file__).parent
rows = [json.loads(line) for line in (p / 'baseline-results.jsonl').read_text().splitlines()]
assert len(rows) == 10, f'Baseline incomplete: {len(rows)}/10 requests'
lines = ['# GPU Station unchanged baseline', '',
    'Measured from the Mac over Tailscale using the OpenAI-compatible streaming endpoint. Qwen3 8B, 32K context, default thinking enabled, temperature 0.2, maximum output 2048 tokens. No persistent configuration changes.', '',
    'TTFT counts the first nonempty reasoning or answer delta. First answer counts the first nonempty content delta. Role-only chunks are excluded. One cold sample; warm values are medians of three repetitions.', '',
    '| Condition / prompt | TTFT | First answer | Total |',
    '|---|---:|---:|---:|']
for state, prompt in [('cold','ping'), ('warm','ping'), ('warm',"what's a diffusion model?"), ('warm','what model are you')]:
    group = [r for r in rows if r['state'] == state and r['prompt'] == prompt]
    values = [statistics.median(r[k] for r in group) for k in ['ttft_s', 'first_answer_s', 'total_s']]
    lines.append(f'| {state}: {prompt} | ' + ' | '.join(f'{v:.2f} s' for v in values) + ' |')
warm = [r for r in rows if r['state']=='warm']
lines += ['', f'Warm TTFT range: {min(r["ttft_s"] for r in warm):.2f}–{max(r["ttft_s"] for r in warm):.2f} seconds.', '',
    'These are standalone prompts, with no OpenCode system prompt, tools, or conversation history. They do not reproduce the full Build workload. Warm requests may reuse prompt cache. The initial cold sample explicitly unloaded only this model first. All requests should end with finish_reason=stop; raw usage counts and timing are in baseline-results.jsonl.', '',
    'The original OpenCode logs showed roughly 10.7K input tokens, 63 tokens/second decoding, and 32.11 / 11.27 / 8.35 seconds server request duration. The UI durations supplied by the user were 31.6 / 11.0 / 9.9 seconds, with 7.8 / 8.8 / 7.6 seconds thinking. Neither UI total duration nor thinking duration is a direct TTFT measurement.', '',
    'Re-run baseline.py only after archiving the current results: it overwrites baseline files. Use the same prompts, sampling parameters, client path, and cold/warm procedure when comparing improvements.']
assert all(r['finish_reason']=='stop' for r in rows)
report = '\n'.join(lines) + '\n'
(p / 'baseline.md').write_text(report)
print(report)
