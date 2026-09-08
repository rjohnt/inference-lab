# Working agreement

This repository is the canonical home for the user's inference, GPU profiling,
and kernel experiments. Each exercise gets one directory under `exercises/`.
Keep its README, runnable code, configs, compact results, and charts together.
Embed generated comparison charts in the exercise README using relative PNG
image links so GitHub displays them inline; retain links to SVG versions.
Use `src/`, `configs/`, `results/`, and `analysis/` as useful; do not split a new
exercise's report across unrelated top-level directories.

Develop here, execute on the configured target machine, collect reviewed results,
then commit and push completed checkpoints to this repository as authorized by
the user. Preserve failed experiments and limitations in the exercise README;
never label a setup check as a measured performance improvement.

Before committing or pushing, follow SECURITY.md, run Gitleaks and the pre-commit
hooks, and inspect the staged files. Never commit credentials, private machine
details, local connection configuration, raw requests/responses, or unreviewed
logs. Pass sensitive configuration through the environment or ignored local files.
Model weights, virtual environments, compiler caches, and large traces stay out
of Git. Retain original captures privately when importing reviewed summaries.

For kernel experiments, compare eager PyTorch, torch.compile, and custom kernels;
verify numerical correctness and separate compilation/warmup from steady-state
timing. Do not promise a speedup before measuring it.
