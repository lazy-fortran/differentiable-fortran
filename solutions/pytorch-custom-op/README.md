# PyTorch custom operator

This solution registers a native CPU operator that receives contiguous PyTorch
`float64` tensors and calls the ISO C Fortran library. A second native operator
provides the analytical VJP, and `torch.library.register_autograd` connects it to
PyTorch reverse-mode differentiation.

```console
solutions/pytorch-custom-op/build.sh
```

The operator is visible to `torch.compile`, so the steady-state execution does not
make an ordinary Python foreign-function call. Like JAX FFI, it remains an opaque
compiled operation and cannot fuse through the Fortran kernel. The implementation
shown here is CPU-only.
