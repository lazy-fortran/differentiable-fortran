# Solution matrix

The derivative engine and consumer adapter are independent choices. “Explicit”
means the adapter exposes a derivative entry point supplied by analytical Fortran
or Enzyme; “framework” means the derivative is generated from framework-native
operations.

| Solution | Primal | JVP | VJP | Boundary used by the consumer |
|---|---:|---:|---:|---|
| Analytical Fortran | yes | analytical | analytical | Fortran or ISO C |
| Flang + Enzyme | yes | generated | generated | ISO C |
| LFortran + Enzyme | yes | generated | generated | ISO C |
| ISO C + `ctypes` | yes | explicit | explicit | Python foreign-function call |
| f90wrap object API | yes | explicit method | not exposed | generated Python extension |
| Cython adapter | yes | not exposed | not exposed | compiled Python extension |
| SciPy `LowLevelCallable` | scalar callback | not exposed | not exposed | direct callback inside SciPy |
| NumPy ufunc | scalar point update | not exposed | not exposed | compiled NumPy loop |
| Python/NumPy Autograd | yes | framework | framework | eager Python tracing and NumPy operations |
| JAX native | yes | framework | framework | JAX primitives |
| JAX FFI | yes | explicit | explicit | XLA FFI handler |
| PyTorch native | yes | framework | framework | PyTorch operations |
| PyTorch custom operator | yes | not exposed | explicit autograd rule | registered native operator |
| Tesseract | yes | explicit | explicit | Tesseract endpoint |

A dash means that the adapter example does not register that operation. The same
shared Fortran library may still export it through the ISO C API. This distinction
keeps a narrow interface example from implying unsupported framework
transformations.
