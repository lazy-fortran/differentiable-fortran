# Limitations

This repository compares one regular, allocation-free stencil. It demonstrates
toolchain and interface mechanics but does not establish that the same ranking
holds for a large multiphysics application, irregular control flow, dynamic
polymorphism inside differentiated regions, distributed memory, or accelerators.

The object-oriented Fortran example exposes a derived type with private state and
type-bound procedures. The minimal ISO C kernel supplies the narrow boundary
required by C-compatible tools. A production library can expose both APIs.

JAX FFI and PyTorch custom operators are opaque compiled calls unless derivative
rules and shape behavior are registered explicitly. They may prevent fusion across
the call. Their value is avoiding Python in the steady-state execution path, not
making every small foreign call free.

Tesseract service timings include effects that are irrelevant to an in-process
numerical call. They are reported separately to prevent orchestration overhead from
being mistaken for derivative cost.
