"""Render a self-contained report from complete, comparable benchmark runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rmsnorm-benchmark"
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for path in args.runs:
        assert (path / "COMPLETE").exists(), f"Incomplete run: {path.name}"
        manifest = json.loads((path / "manifest.json").read_text())
        data = json.loads((path / "summary.json").read_text())
        expected = len(manifest["config"]["rows"]) * len(manifest["config"]["widths"]) * len(manifest["config"]["dtypes"])
        assert len(data) == expected and len({row["case"] for row in data}) == expected
        if runs:
            assert manifest["config_sha256"] == runs[0][1]["config_sha256"], "Different benchmark configurations"
            assert manifest["source_sha256"] == runs[0][1]["source_sha256"], "Different source snapshots"
        runs.append((path.name, manifest, {row["case"]: row for row in data}))
    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted(runs[0][2], key=lambda k: (runs[0][2][k]["dtype"], runs[0][2][k]["rows"], runs[0][2][k]["width"]))
    flat = []
    for name, manifest, data in runs:
        for key in keys:
            row = data[key]
            for metric, methods in row["metrics"].items():
                for method in manifest["config"]["methods"]:
                    ratio = methods["versus_eager"].get(method, {"speedup": 1, "ci95_low": 1, "ci95_high": 1})
                    flat.append({"run": name, "case": key, "metric": metric, "method": method,
                                 **methods[method], **ratio})
    with (args.out / "timings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    methods = [m for m in runs[0][1]["config"]["methods"] if m != "eager"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    y = np.arange(len(keys))
    count = len(runs) * len(methods)
    height = 0.76 / count
    for ax, metric, title in zip(axes, ["graph_gpu", "wall"],
                                 ["GPU execution · CUDA Graphs", "Python calls · synchronized batches"]):
        index = 0
        for name, manifest, data in runs:
            for method in methods:
                values = [data[k]["metrics"][metric]["versus_eager"][method]["speedup"] for k in keys]
                ax.barh(y - 0.38 + height / 2 + index * height, values, height,
                        label=f"{name}: {method}")
                index += 1
        ax.axvline(1, color="#333333", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Speedup over eager (higher is better)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, keys)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Residual + weighted RMSNorm · RTX 4070 SUPER", fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out / "speedups.png", dpi=160, metadata={"Software": "Matplotlib"})
    fig.savefig(args.out / "speedups.svg", metadata={"Date": None})
    svg = args.out / "speedups.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    env = runs[0][1]["environment"]
    cfg = runs[0][1]["config"]
    text = ["# Eager versus compiled residual + RMSNorm", "",
            "## Question and prediction", "",
            "Does Inductor compilation reduce warmed execution time for residual addition, RMS normalization, and learned scaling? Fewer launches and intermediate memory transfers should help, particularly on small workloads.", "",
            "## Workload and environment", "",
            f"{env['gpu']}; PyTorch {env['torch']}; Triton {env['triton']}; CUDA {env['cuda_runtime']}; driver {env['gpu_initial']['driver_version']}; Python {env['python']}. CPU: {env['cpu']}. Linux/WSL, one PyTorch CPU thread. GPU clocks were not locked.", "",
            "Forward inference only; contiguous row-major inputs; FP32 and BF16; residual addition rounds to input dtype before normalization; FP32 accumulation and scaling; output returns to input dtype. Only the normalized output is returned. This includes residual addition and weight scaling, not standalone RMSNorm.", "",
            "## Method", "",
            f"{len(runs)} independent process runs, {len(keys)} cases each, seed {cfg['seed']}. Each method receives the same inputs, {cfg['warmup_calls']} warmup calls, and {cfg['rounds']} measured rounds. Method order is randomized with balanced first/second positions; case order is deterministically shuffled. Inputs and allocations are reused across rounds (warm-cache microbenchmark).", "",
            f"GPU timing: CUDA events around {cfg['graph_replays_per_round']} replays of a graph containing {cfg['graph_batch_calls']} calls, divided by total calls. Python timing: synchronized wall-clock batch of {cfg['wall_batch_calls']} ordinary calls, divided by batch size. Graph capture and compilation are excluded from both. Automatic compiler CUDA Graphs are disabled so graph treatment is matched between implementations.", "",
            "Median and p95 below are distributions of **per-call batch averages**, not individual-request tail latency. Bootstrap confidence intervals in summaries resample paired rounds and describe within-run variability only. First-call durations are saved separately and may hit existing compiler caches; they are not cold compilation benchmarks.", "",
            "Correctness: three random input seeds plus zeros for every method/shape, checked against an independent FP64 reduction with explicit dtype tolerances; input immutability, output dtype/shape, and captured-graph output are checked. These tests cover the benchmark inputs, not all possible numerical extremes.", "",
            "## Results", "", "![Speedups](speedups.svg)", ""]
    for name, manifest, data in runs:
        text += [f"### {name}", "", "| Case | Method | GPU eager µs p50 / p95 | GPU method µs p50 / p95 | GPU speedup | Python eager µs p50 / p95 | Python method µs p50 / p95 | Python speedup |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for key in keys:
            g = data[key]["metrics"]["graph_gpu"]
            w = data[key]["metrics"]["wall"]
            def latency(v):
                return f"{v['p50_us']:.2f} / {v['p95_us']:.2f}"
            for method in methods:
                text.append(f"| {key} | {method} | {latency(g['eager'])} | {latency(g[method])} | {g['versus_eager'][method]['speedup']:.2f}× | {latency(w['eager'])} | {latency(w[method])} | {w['versus_eager'][method]['speedup']:.2f}× |")
        text.append("")
    profile_path = args.out / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        assert profile["implementation_sha256"] == runs[0][1]["source_sha256"]["implementations.py"]
        text += ["## Profile evidence", "", "Separate profiling of one warmed call per case (outside timed rounds):", "",
                 "| Shape / dtype | Method | GPU kernels |", "|---|---|---:|"]
        for row in profile["profiles"]:
            text.append(f"| {row['rows']} × {row['width']} / {row['dtype']} | {row['method']} | {row['kernel_count']} |")
        text += ["", "[Kernel names and counts](profile.json). This supports the launch-fusion explanation; no hardware-counter measurement of DRAM traffic was performed.", ""]
    text += ["## Interpretation and limitations", "",
             "These measurements establish a local baseline for later custom kernels. The graph result isolates device execution more closely; the Python result also reflects dispatch, allocation and synchronization overhead. Neither is end-to-end model throughput. Reused inputs may fit in GPU cache for small shapes; do not extrapolate these ratios to streaming-memory workloads.", "",
             "The second process run checks immediate repeatability on the same machine, not cross-day or cross-machine reproducibility. Windows display activity, thermal state, power management, and unlocked clocks can affect results. Per-case GPU snapshots are retained in summaries. Do not average the two timing modes together.", "",
             "## Reproduction and later optimizations", "",
             "From the exercise directory on the GPU host:", "", "```bash", "source env.sh",
             "python src/benchmark.py --config configs/baseline.json --out local/new-run1",
             "python src/benchmark.py --config configs/baseline.json --out local/new-run2",
             "python analysis/report.py local/new-run1 local/new-run2 --out local/new-report", "```", "",
             "Output directories must be new. `COMPLETE` is written only after all cases pass. Raw numeric samples stay in ignored `local/`; reviewed manifests, summaries, CSV and charts can be published. Source/configuration SHA256 values identify the measured experiment. Add later kernels through `implementations.build`, keep the operation and protocol fixed, and include both eager and compiled baselines in each new run.", ""]
    (args.out / "report.md").write_text("\n".join(text))
    print(f"Rendered {len(runs)} runs, {len(keys)} cases per run")


if __name__ == "__main__":
    main()
