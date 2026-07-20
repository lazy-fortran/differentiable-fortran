# Cython adapter

Cython supplies a compiled, typed adapter from NumPy arrays to the same ISO C
symbol used by `ctypes`. It validates a contiguous `float64` memoryview, allocates
the output, releases the GIL, and calls Fortran without Python argument conversion
inside the native call.

```console
solutions/cython/build.sh
```

The compiled adapter usually has lower boundary overhead than a general `ctypes`
call and can do more work in Cython before returning to Python. For large kernels,
both interfaces should converge toward the same Fortran runtime because boundary
overhead becomes negligible.
