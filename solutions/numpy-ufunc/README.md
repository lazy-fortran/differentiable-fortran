# NumPy ufunc

This solution turns the scalar Fortran stencil point into a genuine NumPy ufunc
with a compiled Cython loop. NumPy broadcasts the four inputs and invokes the
compiled loop without a Python call per element.

```console
solutions/numpy-ufunc/build.sh
```

The ufunc expresses the interior point update; the two copied boundary values are
assembled outside it. This is useful when a Fortran scalar operation should
participate in NumPy broadcasting. A whole-array Fortran kernel remains the more
natural and usually faster representation for this stencil.
