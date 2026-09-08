"""Paired, deterministic RMSNorm benchmark. Raw samples stay in ignored local/."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
import triton

from implementations import build, oracle

ROOT = Path(__file__).resolve().parents[1]
SMI = "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "nvidia-smi"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state():
    fields = ["driver_version", "memory.used", "utilization.gpu", "temperature.gpu",
              "clocks.sm", "clocks.mem", "power.draw", "power.limit"]
    p = subprocess.run([SMI, "--query-gpu=" + ",".join(fields),
                        "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True, check=True)
    return dict(zip(fields, [x.strip() for x in p.stdout.strip().split(",")]))


def inputs(rows, width, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, width, dtype=dtype, device="cuda", generator=generator),
            torch.randn(rows, width, dtype=dtype, device="cuda", generator=generator),
            torch.randn(width, dtype=dtype, device="cuda", generator=generator))


def validate(fn, rows, width, dtype, cfg):
    checks = []
    for seed in cfg["correctness_seeds"] + [None]:
        data = inputs(rows, width, dtype, seed if seed is not None else 0)
        if seed is None:
            data[0].zero_()
            data[1].zero_()
        originals = tuple(x.clone() for x in data)
        expected = oracle(*data, cfg["epsilon"])
        actual = fn(*data)
        assert actual.shape == data[0].shape and actual.dtype == dtype
        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, expected, **cfg["tolerances"][str(dtype).split(".")[-1]])
        for before, after in zip(originals, data):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        difference = (actual.float() - expected.float()).abs()
        checks.append({"seed": seed, "max_abs_error": difference.max().item(),
                       "rms_error": difference.square().mean().sqrt().item()})
    return checks


def capture(fn, data, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn(*data)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            output = fn(*data)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output  # Keep the final output allocation alive through replay.


def measure_wall(fn, data, calls):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(calls):
        output = fn(*data)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1000 / calls


def measure_graph(graph, cfg):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(cfg["graph_replays_per_round"]):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (cfg["graph_replays_per_round"] * cfg["graph_batch_calls"])


def stats(values):
    return {"p50_us": float(np.median(values)), "p95_us": float(np.percentile(values, 95)),
            "min_us": float(min(values)), "max_us": float(max(values)),
            "rounds": len(values)}


def ratio_stats(a, b, seed):
    # Paired bootstrap of rounds estimates within-run uncertainty, not day-to-day drift.
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(a), size=(2000, len(a)))
    ratios = np.median(np.asarray(a)[indexes], axis=1) / np.median(np.asarray(b)[indexes], axis=1)
    return {"speedup": float(np.median(a) / np.median(b)),
            "ci95_low": float(np.percentile(ratios, 2.5)),
            "ci95_high": float(np.percentile(ratios, 97.5))}


def run_case(case, cfg, rng, samples_file):
    rows, width, dtype_name = case
    label = f"{dtype_name}-{rows}x{width}"
    dtype = getattr(torch, dtype_name)
    before = gpu_state()
    data = inputs(rows, width, dtype, cfg["seed"])
    functions, graphs, checks, first_call = {}, {}, {}, {}
    for name in cfg["methods"]:
        fn = build(name, cfg["epsilon"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*data)
        torch.cuda.synchronize()
        first_call[name] = time.perf_counter() - start
        checks[name] = validate(fn, rows, width, dtype, cfg)
        for _ in range(cfg["warmup_calls"]):
            fn(*data)
        torch.cuda.synchronize()
        functions[name] = fn
        graphs[name] = capture(fn, data, cfg["graph_batch_calls"])
        # Check graph replay, not just its uncaptured implementation.
        torch.testing.assert_close(graphs[name][1], oracle(*data, cfg["epsilon"]),
                                   **cfg["tolerances"][dtype_name])
    samples = {metric: {name: [] for name in cfg["methods"]}
               for metric in ["graph_gpu", "wall"]}
    # Equal first/second position counts, with randomized paired order per metric.
    orders = {}
    for metric in samples:
        orders[metric] = [cfg["methods"][i % len(cfg["methods"]):] + cfg["methods"][:i % len(cfg["methods"]) ]
                          for i in range(cfg["rounds"])]
        rng.shuffle(orders[metric])
    for index in range(cfg["rounds"]):
        metrics = list(samples)
        rng.shuffle(metrics)
        for metric in metrics:
            for name in orders[metric][index]:
                if metric == "wall":
                    elapsed = measure_wall(functions[name], data, cfg["wall_batch_calls"])
                else:
                    elapsed = measure_graph(graphs[name][0], cfg)
                samples[metric][name].append(elapsed)
                samples_file.write(json.dumps({"case": label, "round": index,
                                               "metric": metric, "method": name, "us": elapsed}) + "\n")
        samples_file.flush()
    result = {"case": label, "rows": rows, "width": width, "dtype": dtype_name,
              "correctness": checks, "first_call_seconds_including_compile": first_call,
              "gpu_before": before, "gpu_after": gpu_state(), "metrics": {}}
    for metric, methods in samples.items():
        result["metrics"][metric] = {name: stats(v) for name, v in methods.items()}
        result["metrics"][metric]["versus_eager"] = {
            name: ratio_stats(methods["eager"], v, cfg["seed"])
            for name, v in methods.items() if name != "eager"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--methods", nargs="+", help="Override registry implementations; include eager")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.methods:
        cfg["methods"] = args.methods
    assert "eager" in cfg["methods"] and len(set(cfg["methods"])) == len(cfg["methods"])
    assert cfg["rounds"] >= 10
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    info = torch.cuda.get_device_properties(0)
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unknown")
    memory_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemTotal:")))
    manifest = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "src").glob("*.py"))},
                "environment": {"python": platform.python_version(), "os": platform.system(),
                                "kernel": platform.release(), "cpu": cpu_model, "ram_kib": memory_kib,
                                "torch": torch.__version__, "triton": triton.__version__,
                                "cuda_runtime": torch.version.cuda, "gpu": info.name,
                                "vram_bytes": info.total_memory, "multiprocessors": info.multi_processor_count,
                                "compute_capability": [info.major, info.minor], "torch_cpu_threads": 1,
                                "gpu_initial": gpu_state()},
                "compiler": {"backend": "inductor", "fullgraph": True, "dynamic": False,
                             "triton.cudagraphs": False, "cache": "persistent; first call excluded"}}
    save(args.out / "manifest.json", manifest)
    cases = [(r, w, d) for r in cfg["rows"] for w in cfg["widths"] for d in cfg["dtypes"]]
    rng = random.Random(cfg["seed"])
    rng.shuffle(cases)
    manifest["case_order"] = cases
    save(args.out / "manifest.json", manifest)
    results = []
    with (args.out / "samples.jsonl").open("x") as samples_file, torch.inference_mode():
        for case in cases:
            result = run_case(case, cfg, rng, samples_file)
            results.append(result)
            save(args.out / "summary.json", results)
            print(json.dumps({"completed": len(results), "total": len(cases), "case": result["case"],
                              "speedups": {metric: value["versus_eager"]
                                           for metric, value in result["metrics"].items()}}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.out / "manifest.json", manifest)
    (args.out / "COMPLETE").write_text(f"{len(results)} cases; all correctness checks passed\n")


if __name__ == "__main__":
    main()
