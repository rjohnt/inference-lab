# Ollama streaming baseline

Measured cold and warm streaming latency for Qwen3 8B. Warm TTFT medians ranged from 0.29 to 3.49 seconds across the three tested prompts; one cold ping took 13.35 seconds to first token. These are small-sample observations, not a general serving benchmark.

[Baseline report](notes/baseline.md)

Imported September 8, 2026 from the completed local experiments.

- `src/`: preserved experiment and analysis scripts.
- `notes/`: historical setup notes, findings, and reports.
- `results/`: reviewed summaries, configuration metadata, and SVG charts.

These are historical imports, not fresh reruns. Machine-specific paths and addresses were replaced with examples. Configure paths and endpoints before running the historical scripts; do not execute administration scripts without reviewing their host changes. Raw requests, generated responses, logs, model weights, build trees, and personal configuration remain outside Git. Some historical report links refer to those local captures.

The source scripts generate new raw outputs; keep these in ignored local storage and publish only reviewed summaries. Model revisions and checksums are retained where captured.
