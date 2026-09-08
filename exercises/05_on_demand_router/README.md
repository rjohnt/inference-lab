# On-demand authenticated model routing

Preserves authenticated, serialized model switching, 120-second idle unloading, streaming checks, and wake-after-idle validation. This is an operational experiment; cold load, queue time, and warm inference latency are different metrics.

[Behavior and validation](notes/README.md)

Set `GPUSTATION_KEY_FILE` to a nonempty credential file outside Git. Check scripts also require `INFERENCE_BASE_URL`. Service files and model presets are examples requiring local paths. No credential value is included.

Imported September 8, 2026 from the completed local experiments.

- `src/`: preserved experiment and analysis scripts.
- `notes/`: historical setup notes, findings, and reports.
- `results/`: reviewed summaries, configuration metadata, and SVG charts.

These are historical imports, not fresh reruns. Machine-specific paths and addresses were replaced with examples. Configure paths and endpoints before running the historical scripts; do not execute administration scripts without reviewing their host changes. Raw requests, generated responses, logs, model weights, build trees, and personal configuration remain outside Git. Some historical report links refer to those local captures.

The source scripts generate new raw outputs; keep these in ignored local storage and publish only reviewed summaries. Model revisions and checksums are retained where captured.
