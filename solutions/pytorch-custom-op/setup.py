from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parents[2]

setup(
    name="df-torch-operator",
    ext_modules=[
        CppExtension(
            "_df_torch_operator",
            [str(Path(__file__).with_name("operator.cpp"))],
            libraries=["differentiable_fortran"],
            library_dirs=[str(ROOT / "build")],
            runtime_library_dirs=[str(ROOT / "build")],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
