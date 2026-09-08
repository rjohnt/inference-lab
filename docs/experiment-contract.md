# Experiment contract

Every exercise is a small scientific and engineering artifact. Use this contract
to make results credible to a reviewer who does not have the original machine.

## Required repository layout

```text
exercises/<stage>-<slug>/
  README.md             # question, setup, and commands
  src/                  # runnable implementation
  configs/              # tracked workload/configuration files
  results/              # compact CSV/JSON summaries only
  analysis/             # chart or analysis code
  report.md             # final write-up, or include it in README.md
  charts/               # reviewed, shareable visual results
```

Raw profiler captures and model weights belong outside Git. Store a stable
reference to their location or acquisition command in the report when needed.

## Before running

- State the question, primary metric, and success/failure condition.
- State a prediction: expected bottleneck and expected direction of change.
- Record the independent variables, controls, warm-up policy, repetition count,
  synchronization method, and random seed where relevant.
- Capture the environment: GPU, CPU, RAM, OS, driver, CUDA, Python, PyTorch,
  serving engine, model revision, dtype, and relevant runtime settings.

## Measurement rules

- Separate warm-up from measured iterations.
- Synchronize asynchronous GPU work before timing it.
- Report median and tail latency (at least p95), not only a mean.
- Pair latency with throughput and memory usage when evaluating serving systems.
- Use the same request mix and generation parameters when comparing configurations.
- Keep raw structured results locally; commit compact, derived tables and charts.

## Report outline

Copy this outline into the exercise README or `exercises/<stage>-<slug>/report.md`:

```markdown
# Title

## Question and prediction
## Workload and environment
## Method
## Results
## Profile evidence
## Interpretation
## Change tested and validation
## Recommendation and limitations
## Reproduction
```

The key standard: a reader should be able to distinguish what was measured, what
was observed in a profile, and what is an inference or recommendation.
