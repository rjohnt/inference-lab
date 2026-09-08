# Nsight Compute: explain RMSNorm fusion

Status: **prepared for a future profiling session; no Nsight Compute measurements
have been collected for this exercise.** The sustained timing benchmark must
finish before profiling begins.

## Question and prediction

Why is compiled residual + weighted RMSNorm faster on the GPU, and what limits
further improvement? The timing exercise found 10 eager kernels versus one
compiled kernel in its BF16 profiler checks. This experiment will test whether
fusion reduces DRAM traffic, how occupancy changes, and whether the large case
is bandwidth limited. No Nsight Compute counter result is claimed yet.

The real workload is [`src/profile_workload.py`](src/profile_workload.py). It
imports the exact operation from the sibling kernel-fusion exercise, checks
correctness, performs 100 warmup calls, and places **one invocation** inside the
`rmsnorm` NVTX range. It does not run the multi-million-call timing sweep under
the profiler. Kernel source hashes are printed for provenance.

## Run later on the GPU host

Keep both exercise directories together in an Inference Lab checkout. Prepare
the sibling `04_kernel_fusion/.venv` with its setup script. `python3` below runs
only the standard-library launcher; the workload is launched using the absolute
path of the CUDA-enabled virtual-environment Python.

From this directory:

```bash
ncu --version
ncu --help
python3 src/run_ncu.py --method eager --set basic --out local/eager-basic
python3 src/run_ncu.py --method compiled --set basic --out local/compiled-basic

# Once the basic reports succeed:
python3 src/run_ncu.py --method eager --set full --out local/eager-full
python3 src/run_ncu.py --method compiled --set full --out local/compiled-full
ncu-ui local/compiled-full/report.ncu-rep
```

Defaults: BF16, 4,096 rows × 4,096 columns, seed 20260908, epsilon 1e-6. This is
the same activation shape as batch 8 × 512 token positions with hidden width
4,096. Repeat later with `--rows 1` for the small-workload contrast. Use new output
directories for repeated captures.

For an existing standalone kernel installation, set `RMSNORM_EXERCISE_DIR` to
that directory in your local shell, or pass `--kernel-exercise`. `--python` and
`--ncu` accept explicit executable paths. No machine-specific values belong in
Git. `--dry-run` prints the resolved command without executing anything.

## Collection choices

The launcher uses launch-and-attach and an NVTX push/pop range filter. Compilation,
allocation, warmup and verification are outside that range. It collects all
kernels for the marked invocation: using a one-kernel launch limit would omit
most of the eager operation. Child-process coverage is enabled explicitly.

Clock control and cache flushing are disabled explicitly in this initial recipe;
record those choices when interpreting the report. Kernel replay still perturbs
execution. Treat hardware-counter profiling as an explanation tool, and use the
separate unprofiled benchmark for performance claims. Do not run the two together.
See NVIDIA's [CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Metrics to compare

| Question | Evidence to inspect |
|---|---|
| Was the operation fused? | Kernel names and launch count across the marked invocation |
| Did fusion remove intermediate traffic? | DRAM read/write bytes, plus L2 traffic and hit rates |
| Is execution bandwidth limited? | Memory throughput relative to sustained peak; Speed of Light sections |
| Does the launch configuration limit utilization? | Grid/block dimensions, achieved occupancy, registers/thread, occupancy limits |
| What prevents issue progress? | Scheduler/warp-stall analysis where available |
| What should a custom kernel change? | The measured bottleneck, checked again with unprofiled timing |

Sum comparable byte counters over all eager kernels when comparing a complete
operation. Do not compare the fused kernel against only one eager sub-kernel.
Do not sum occupancy percentages; interpret those per kernel. Counter and section
names vary with profiler version and GPU, so inspect the installed tool's sets.

## Why the earlier commands failed

- `python` was not an available executable; the launcher resolves the venv Python.
- `your_script.py`, `train.p`, and `train.py` were nonexistent example filenames.
  This directory supplies an actual workload and resolves its absolute path.
- A process exiting before CUDA work yields no kernels to profile.
- An unsupported `--process-id` flag is not proof that arbitrary-process attach
  would work with an upgrade. NVIDIA's documented attach workflow requires a
  target previously started under Nsight Compute's launch instrumentation.
  Launching the workload directly with `ncu` avoids that ambiguity.
- If profiling later reports `ERR_NVGPUCTRPERM`, follow NVIDIA's
  [performance-counter permission instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
  This recipe does not automatically alter driver permissions.

## Publish after execution

Keep `.ncu-rep`, launch logs, and raw metadata under ignored `local/`; they can
contain paths and process identifiers. Publish reviewed counter tables, a report,
and SVG/PNG comparisons inside this exercise after credential/machine-detail
review. Record profiler version, GPU/driver, cache/clock settings, shape/dtype,
source hash, and limitations. An empty or failed capture is not a completed study.
