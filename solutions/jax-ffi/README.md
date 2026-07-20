# JAX FFI

This solution registers three CPU XLA FFI targets: the primal, analytical JVP, and
analytical VJP. Each target is a thin C++ handler that receives XLA buffers and
calls the ISO C Fortran library. A `jax.jit`-compiled program executes the handler
without returning to Python.

```console
solutions/jax-ffi/build.sh
```

The steady-state call can be fast because Python is absent from the compiled path.
It remains an opaque XLA operation, so XLA cannot fuse through the Fortran call.
The handler shown here is CPU-only; GPU execution requires a GPU handler and a
Fortran or other compiled kernel that runs on that device. Derivatives are explicit
FFI targets because JAX cannot infer a derivative through foreign code.

The plain `jvp` and `vjp` functions make those derivative targets visible and easy
to benchmark. A production JAX API can attach them as transformation rules using
the JAX custom-derivative or HiJAX APIs appropriate to its supported JAX version.
