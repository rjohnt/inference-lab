import torch

import gpu_profiling_ext


def main() -> None:
    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    x = torch.arange(1_000_000, device="cuda", dtype=torch.float32)
    y = gpu_profiling_ext.add_one(x)
    torch.cuda.synchronize()
    assert y.is_cuda
    assert torch.equal(y, x + 1)
    print("PASS: compiled C++ extension accepted and returned CUDA tensors")


if __name__ == "__main__":
    main()
