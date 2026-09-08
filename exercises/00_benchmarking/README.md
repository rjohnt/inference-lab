# 00 — Measurement foundations

## Objective

Build the reusable benchmark harness that every later exercise will use. Its first
workloads are matrix multiplication, softmax, normalization, and host-to-device
copies across tensor sizes and available dtypes.

## Questions

1. Which workloads are compute-bound or memory-bound at different sizes?
2. How do FP32, FP16, and BF16 change latency, throughput, memory, and utilization?
3. What measurement mistakes would make these conclusions unreliable?

## First implementation checklist

- [ ] Make a Linux GPU environment manifest command.
- [ ] Implement synchronized timing with warm-up and repeated samples.
- [ ] Emit machine-readable per-run results (CSV or JSON).
- [ ] Add matrix-size and dtype sweep configs.
- [ ] Capture peak allocated/reserved memory separately where applicable.
- [ ] Produce latency and throughput charts from the result files.
- [ ] Write `reports/00-benchmarking.md` using the experiment contract.

## Initial interface proposal

Keep the harness simple and explicit:

```text
python -m src.run --workload matmul --config configs/matmul.yaml
python -m analysis.plot --input results/matmul.jsonl --output ../../artifacts/00-benchmarking/
```

The implementation will be added once the Linux CUDA environment is available.
