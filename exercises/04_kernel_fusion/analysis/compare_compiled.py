"""Direct custom-kernel comparison against the co-run compiler control."""
import argparse
import csv
import json
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
    rows = sorted(json.loads((args.run / "summary.json").read_text()), key=lambda r: (r["dtype"], r["rows"], r["width"]))
    args.out.mkdir(parents=True, exist_ok=True)
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    for ax, metric, title in zip(axes, ("graph_gpu", "wall"), ("GPU execution · CUDA Graph replay", "Ordinary Python calls · synchronized batches")):
        ratios = []
        for r in rows:
            m = r["metrics"][metric]
            ratio = m["compiled"]["p50_us"] / m["triton_fused"]["p50_us"]
            ratios.append(ratio)
            table.append({"case": r["case"], "metric": metric,
                          "compiled_p50_us": m["compiled"]["p50_us"],
                          "triton_p50_us": m["triton_fused"]["p50_us"],
                          "triton_speedup_over_compiled": ratio})
        ax.barh(np.arange(len(rows)), ratios, color=["#238b6d" if v>1 else "#d87838" for v in ratios])
        for i, v in enumerate(ratios):
            ax.text(v + .02, i, f"{v:.2f}×", va="center", fontsize=9)
        ax.axvline(1, color="#334155", linestyle="--")
        ax.set_xlim(0, max(ratios) * 1.18)
        ax.set_title(title); ax.set_xlabel("Compiled latency / custom Triton latency (higher is better)")
        ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [r["case"] for r in rows]); axes[0].invert_yaxis()
    fig.suptitle("Does the handwritten kernel beat torch.compile?\nCo-run controls · residual + weighted RMSNorm", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out / "versus-compiled.png", dpi=150)
    fig.savefig(args.out / "versus-compiled.svg", metadata={"Date": None})
    p = args.out / "versus-compiled.svg"
    p.write_text("\n".join(s.rstrip() for s in p.read_text().splitlines()) + "\n")
    with (args.out / "versus-compiled.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(table)
    print(json.dumps({metric: {"min_speedup": min(r["triton_speedup_over_compiled"] for r in table if r["metric"] == metric),
                               "max_speedup": max(r["triton_speedup_over_compiled"] for r in table if r["metric"] == metric),
                               "wins": sum(r["triton_speedup_over_compiled"] > 1 for r in table if r["metric"] == metric)} for metric in ("graph_gpu", "wall")}))


if __name__ == "__main__":
    main()
