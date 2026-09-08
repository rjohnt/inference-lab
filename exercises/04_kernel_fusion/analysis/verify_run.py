"""Audit a complete raw run against its compact summary, without a GPU."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(path):
    assert (path / "COMPLETE").exists(), "Missing completion marker"
    manifest = json.loads((path / "manifest.json").read_text())
    cfg = manifest["config"]
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == manifest["config_sha256"]
    expected_cases = {f"{d}-{r}x{w}" for r in cfg["rows"] for w in cfg["widths"] for d in cfg["dtypes"]}
    summaries = json.loads((path / "summary.json").read_text())
    assert len(summaries) == len(expected_cases)
    assert {row["case"] for row in summaries} == expected_cases
    groups = defaultdict(dict)
    duration = defaultdict(float)
    call_counts = defaultdict(int)
    raw = (path / "samples.jsonl").read_bytes()
    for line in raw.splitlines():
        sample = json.loads(line)
        key = sample["case"], sample["metric"], sample["method"]
        assert sample["round"] not in groups[key], "Duplicate measured round"
        assert np.isfinite(sample["us"]) and sample["us"] > 0
        groups[key][sample["round"]] = sample["us"]
        if "calls" in sample:
            assert isinstance(sample["calls"], int) and sample["calls"] > 0
            assert np.isclose(sample["batch_us"], sample["us"] * sample["calls"], rtol=1e-12)
            duration[key] += sample["batch_us"] / 1e6
            call_counts[key] += sample["calls"]
    assert len(groups) == len(expected_cases) * 2 * len(cfg["methods"])
    for row in summaries:
        rounds = row.get("actual_rounds", cfg["rounds"])
        assert rounds >= cfg["rounds"]
        assert set(row["correctness"]) == set(cfg["methods"])
        for name, checks in row["correctness"].items():
            assert [c["seed"] for c in checks] == cfg["correctness_seeds"] + [None]
            assert all(np.isfinite(c["max_abs_error"]) and c["max_abs_error"] >= 0 for c in checks)
        for metric in ["graph_gpu", "wall"]:
            for method in cfg["methods"]:
                samples = groups[row["case"], metric, method]
                assert set(samples) == set(range(rounds))
                values = list(samples.values())
                compact = row["metrics"][metric][method]
                assert compact["rounds"] == rounds
                if cfg.get("min_measured_seconds"):
                    key = row["case"], metric, method
                    assert duration[key] >= cfg["min_measured_seconds"], "Measured duration floor not met"
                    assert np.isclose(compact["measured_seconds"], duration[key], rtol=1e-12)
                    assert compact["measured_calls"] == call_counts[key]
                    assert call_counts[key] == rounds * row["calls_per_sample"][metric][method]
                for field, actual in [("p50_us", np.median(values)), ("p95_us", np.percentile(values, 95)),
                                      ("min_us", min(values)), ("max_us", max(values))]:
                    assert np.isclose(compact[field], actual, rtol=1e-12, atol=1e-12)
                if method != "eager":
                    expected = row["metrics"][metric]["eager"]["p50_us"] / compact["p50_us"]
                    assert np.isclose(row["metrics"][metric]["versus_eager"][method]["speedup"], expected)
    return {"status": "PASS", "cases": len(expected_cases), "methods": cfg["methods"],
            "sample_count": sum(len(v) for v in groups.values()),
            "samples_sha256": hashlib.sha256(raw).hexdigest(),
            "checks": ["completion marker", "configuration hash", "case coverage", "round uniqueness",
                       "finite positive samples", "correctness seed coverage", "summary percentiles", "speedup ratios"]
                      + (["minimum rounds", "minimum measured duration", "measured call counts"]
                         if cfg.get("min_measured_seconds") else [])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audit = verify(args.run)
    (args.run / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))
