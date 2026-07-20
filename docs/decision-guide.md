# Decision guide

Choose the derivative engine and the consumer interface separately.

Use an analytical derivative when the formula is maintainable and the kernel is
important enough to justify it. It supplies the strongest oracle and usually the
smallest runtime overhead, but every model change may require a corresponding
derivative change.

Use Flang plus Enzyme when production LLVM tooling and generated derivatives for
Fortran are the priority. Use LFortran plus Enzyme when you want LFortran's
interactive compiler tooling or need to evaluate its generated
IR against Flang on the same source. The benchmark treats these as separate
toolchains, not aliases.

Use native JAX or PyTorch when the model is expressed in that framework and its
transformations or accelerator support determine the implementation.

Use Python/NumPy Autograd as an eager reference when you want readable Python AD
without a compilation phase. Its derivative calls rebuild the trace and tape, so
the benchmark reports it separately from JAX JIT and PyTorch Inductor.

For Python access, use f90wrap to expose modules and derived types with stored
parameters and type-bound procedures. Use a small ISO C ABI as the substrate for
stable cross-language integration. Call it through `ctypes` for coarse-grained
calls, through Cython for a compiled adapter, or through a framework-native FFI
when the call belongs inside a compiled loop.

Tesseract belongs at the packaging and orchestration boundary. It does not replace
the compiler, derivative engine, or Python interface; it packages whichever
combination you selected.
