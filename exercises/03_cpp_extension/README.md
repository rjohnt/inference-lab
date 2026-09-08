# GPU profiling sample

This project builds a minimal C++ PyTorch extension and calls it with a CUDA
tensor. The C++ extension uses PyTorch tensor operators, so it executes on the
GPU without requiring a locally installed CUDA toolkit or `nvcc`.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=1 .venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python test_gpu_extension.py
```

`MAX_JOBS=1` keeps the initial extension compilation modest in memory use.
