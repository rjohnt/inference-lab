"""Inspect early-to-late latency drift in a completed sustained run."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rmsnorm-stability"
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert (args.run / "COMPLETE").exists()
    groups = defaultdict(list)
    with (args.run / "samples.jsonl").open() as stream:
        for line in stream:
            v = json.loads(line)
            groups[v["case"], v["metric"], v["method"]].append((v["round"], v["us"]))
    summary = json.loads((args.run / "summary.json").read_text())
    cases = sorted(summary, key=lambda r: (r["dtype"], r["rows"], r["width"]))
    cfg = json.loads((args.run / "manifest.json").read_text())["config"]
    columns = [(metric, method) for metric in ["graph_gpu", "wall"] for method in cfg["methods"]]
    matrix, table = [], []
    for case in cases:
        row = []
        for metric, method in columns:
            values = np.array([us for _, us in sorted(groups[case["case"], metric, method])])
            count = len(values) // 4
            early = float(np.median(values[:count]))
            late = float(np.median(values[-count:]))
            drift = (late / early - 1) * 100
            row.append(drift)
            table.append({"case": case["case"], "metric": metric, "method": method,
                          "samples": len(values), "quarter_samples": count,
                          "first_quarter_p50_us": early, "last_quarter_p50_us": late,
                          "late_vs_early_percent": drift})
        matrix.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "stability.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    matrix = np.asarray(matrix)
    limit = max(float(np.abs(matrix).max()), 1.0)
    fig, ax = plt.subplots(figsize=(9, 10))
    heatmap = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(cases)), [r["case"] for r in cases])
    ax.set_xticks(range(len(columns)), [f"{metric}\n{method}" for metric, method in columns])
    for i in range(len(cases)):
        for j in range(len(columns)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 0.65 * limit else "black")
    ax.set_title("Sustained-run latency drift\nLast-quarter median relative to first-quarter median")
    fig.colorbar(heatmap, ax=ax, label="Positive = slower at the end (%)")
    fig.tight_layout()
    fig.savefig(args.out / "stability.png", dpi=160, metadata={"Software": "Matplotlib"})
    fig.savefig(args.out / "stability.svg", metadata={"Date": None})
    p = args.out / "stability.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps({"maximum_absolute_drift_percent": float(np.abs(matrix).max()), "comparisons": len(table)}))


if __name__ == "__main__":
    main()
