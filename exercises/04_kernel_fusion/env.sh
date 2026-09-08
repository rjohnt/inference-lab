# Source this file from bash on the GPU host.
export KERNEL_LAB_ROOT="${KERNEL_LAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export TRITON_CACHE_DIR="$KERNEL_LAB_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$KERNEL_LAB_ROOT/.cache/torchinductor"
export CUDA_CACHE_PATH="$KERNEL_LAB_ROOT/.cache/cuda"
export MPLBACKEND=Agg
source "$KERNEL_LAB_ROOT/.venv/bin/activate"
