# September 2026 experiment consolidation

The existing local Inference Lab scaffold had no commits, and the remote had no
branches when checked. This initial publication preserves that scaffold and
imports the accumulated local experiments into separate exercise directories.

| Exercise | Preserved work |
| --- | --- |
| `00_ollama_baseline` | Cold/warm streaming timing harness and baseline report |
| `03_cpp_extension` | The separate GPU profiling project's C++ extension and CUDA check |
| `04_kernel_fusion` | Isolated PyTorch/Triton setup, environment check, experiment design |
| `05_speculative_decoding` | Qwen3 8B no-speculation, n-gram, EAGLE-3 and DFlash runners; historical runs; recovery notes; summaries and charts |
| `05_dflash2_cuda` | Qwen3.8 27B CUDA runner, pinned model download, 120-request report and code review |
| `05_dflash2_metal` | Matched Metal runner, failed probes, model metadata, combined 240-response report and chart |
| `05_on_demand_router` | Authenticated router, launchers, service templates, lifecycle checks and findings |
| `05_single_node_serving` | Existing dependency-free vLLM streaming benchmark |

The imports preserve prior observations, not new executions of those benchmarks.
Historical scripts retain their original relative layout assumptions inside
`src/`; external model paths, endpoints and public-key configuration must be set
for the reader's environment. Some historical commands need path adaptation.
Administration scripts are reference material and were not rerun during import.

Private captures remain at their original local locations. No model weights,
download caches, compiled artifacts, complete request/response dumps, personal
SSH keys, private endpoints, or local client configuration were imported. Public
model revisions, hashes, numerical summaries, reviewed reports, and SVG charts
are retained. Personal paths in historical material are replaced with examples.
Configuration under `configs/` removes local model paths while retaining model
revisions and measurement settings.

`import-manifest.json` lists imported files and SHA256 of their original local
source, allowing comparison against privately retained originals. It is a source
provenance record; hashes are not expected to match sanitized or adapted copies.

Future experiments should be portable from their first commit and keep their
report and outputs inside their exercise directory, as recorded in AGENTS.md.
