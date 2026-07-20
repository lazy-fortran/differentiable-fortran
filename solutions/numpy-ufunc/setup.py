from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[2]

extension = Extension(
    "heat_ufunc",
    [str(Path(__file__).with_name("heat_ufunc.pyx"))],
    include_dirs=[np.get_include(), str(ROOT / "contract")],
    libraries=["differentiable_fortran"],
    library_dirs=[str(ROOT / "build")],
    runtime_library_dirs=[str(ROOT / "build")],
)

setup(
    name="heat-ufunc",
    ext_modules=cythonize([extension], compiler_directives={"language_level": 3}),
)
