# Third-party software

The repository does not vendor the projects below. Build and test environments
install them as tools or runtime dependencies.

| Project | Role | Upstream license |
|---|---|---|
| Enzyme | LLVM automatic differentiation pass | Apache-2.0 with LLVM exceptions |
| Enzyme benchmarks | Mathematical and historical benchmark reference; no source is vendored | No repository-level license; referenced ADBench files are MIT |
| LLVM Flang and LLVM tools | Fortran compiler and LLVM pipeline | Apache-2.0 with LLVM exceptions |
| LFortran | Fortran compiler | BSD-3-Clause |
| f90wrap | Python wrapping of Fortran derived types and modules | LGPL-3.0 |
| Cython | Compiled Python adapters and NumPy ufunc generation | Apache-2.0 |
| NumPy | Arrays and reference implementation | BSD-3-Clause |
| Autograd | Eager Python/NumPy automatic-differentiation baseline | MIT |
| SciPy | `LowLevelCallable` integration example | BSD-3-Clause |
| JAX and jaxlib | Native AD baseline and XLA FFI | Apache-2.0 |
| PyTorch | Native AD baseline and custom operator | BSD-style |
| Tesseract Core | Packaging and orchestration | Apache-2.0 |

Generated wrapper and extension files are build artifacts and are not committed.
Users who distribute binaries containing or linked with these dependencies are
responsible for satisfying the corresponding license notices and conditions.
