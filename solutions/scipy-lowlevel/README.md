# SciPy LowLevelCallable

This example integrates the Fortran `df_gaussian` function with SciPy QUADPACK.
`ctypes` declares the symbol once and `LowLevelCallable` retains its address.
QUADPACK then calls Fortran directly for every integrand evaluation; the adaptive
integration loop does not re-enter Python.

```console
uv run solutions/scipy-lowlevel/example.py
```

This distinction is why “ctypes call overhead” does not describe every program
that mentions `ctypes`. Here it is a setup mechanism for a direct compiled
callback.
