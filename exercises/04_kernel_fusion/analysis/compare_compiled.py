"""Direct custom-kernel comparison against the co-run compiler control."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "triton-compiled-comparison"
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert (args.run / "COMPLETE").exists()
    cfg = json.loads((args.run / "manifest.json").read_text())["config"]
    rows = sorted(json.loads((args.run / "summary.json").read_text()), key=lambda r: (r["dtype"], r["rows"], r["width"]))
    groups = defaultdict(dict)
    sample_path = args.run / "samples.jsonl"
    if sample_path.exists():
        for line in sample_path.read_text().splitlines():
            s = json.loads(line)
            if s["method"] in ("compiled", "triton_fused"):
                groups[s["case"], s["metric"], s["method"]][s["round"]] = s["us"]
    def interval(case, metric):
        if not groups:
            return None
        a, b = [groups[case, metric, name] for name in ("compiled", "triton_fused")]
        assert set(a) == set(b) and set(a) == set(range(len(a)))
        a, b = np.array([a[i] for i in range(len(a))]), np.array([b[i] for i in range(len(b))])
        rng = np.random.default_rng(cfg["seed"])
        block = cfg.get("bootstrap_block_rounds", 1)
        ratios = []
        for _ in range(40):
            starts = rng.integers(0, len(a), size=(50, math.ceil(len(a) / block)))
            idx = ((starts[..., None] + np.arange(block)) % len(a)).reshape(50, -1)[:, :len(a)]
            ratios.extend(np.median(a[idx], axis=1) / np.median(b[idx], axis=1))
        return list(map(float, np.percentile(ratios, [2.5, 97.5])))
    args.out.mkdir(parents=True, exist_ok=True)
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    for ax, metric, title in zip(axes, ("graph_gpu", "wall"), ("GPU execution · CUDA Graph replay", "Ordinary Python calls · synchronized batches")):
        ratios, intervals = [], []
        for r in rows:
            m = r["metrics"][metric]
            ratio = m["compiled"]["p50_us"] / m["triton_fused"]["p50_us"]
            ratios.append(ratio)
            ci = interval(r["case"], metric)
            intervals.append(ci)
            table.append({"case": r["case"], "metric": metric,
                          "compiled_p50_us": m["compiled"]["p50_us"],
                          "triton_p50_us": m["triton_fused"]["p50_us"],
                          "triton_speedup_over_compiled": ratio,
                          "ci95_low": ci[0] if ci else None,
                          "ci95_high": ci[1] if ci else None})
        ax.barh(np.arange(len(rows)), ratios, color=["#238b6d" if v>1 else "#d87838" for v in ratios])
        for i, ci in enumerate(intervals):
            if ci:
                ax.plot(ci, [i, i], color="#172033", linewidth=2)
        for i, v in enumerate(ratios):
            label = f"{v:.3f}×" if abs(v - 1) < .01 else f"{v:.2f}×"
            ax.text(v + .02, i, label, va="center", fontsize=9)
        ax.axvline(1, color="#334155", linestyle="--")
        ax.set_xlim(0, max(ratios) * 1.18)
        ax.set_title(title); ax.set_xlabel("Compiled latency / custom Triton latency (higher is better)")
        ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [r["case"] for r in rows]); axes[0].invert_yaxis()
    note = "paired 95% block-bootstrap intervals" if groups else "point estimates only; raw samples unavailable"
    fig.suptitle("Does the handwritten kernel beat torch.compile?\nCo-run controls · " + note, fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "versus-compiled.png", dpi=150)
    fig.savefig(args.out / "versus-compiled.svg", metadata={"Date": None})
    p = args.out / "versus-compiled.svg"
    p.write_text("\n".join(s.rstrip() for s in p.read_text().splitlines()) + "\n")
    with (args.out / "versus-compiled.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(table)
    (args.out / "versus-compiled.json").write_text(json.dumps({
        "bootstrap_resamples": 2000, "block_rounds": cfg.get("bootstrap_block_rounds", 1),
        "note": "Within-run paired uncertainty only; not cross-day variability. Intervals absent without raw samples.",
        "comparisons": table}, indent=2) + "\n")
    print(json.dumps({metric: {"min_speedup": min(r["triton_speedup_over_compiled"] for r in table if r["metric"] == metric),
                               "max_speedup": max(r["triton_speedup_over_compiled"] for r in table if r["metric"] == metric),
                               "wins": sum(r["triton_speedup_over_compiled"] > 1 for r in table if r["metric"] == metric)} for metric in ("graph_gpu", "wall")}))


if __name__ == "__main__":
    main()
