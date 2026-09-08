from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="gpu_profiling_ext",
    ext_modules=[
        CppExtension(
            name="gpu_profiling_ext",
            sources=["csrc/add_one.cpp"],
            extra_compile_args={"cxx": ["-O0", "-g0"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
