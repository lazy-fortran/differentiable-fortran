# Tesseract packaging and orchestration

This solution packages the LFortran+Enzyme shared library behind Tesseract's
`apply`, JVP, VJP, and abstract-evaluation endpoints. The mathematical contract and
generated derivatives are the same ones used by the in-process comparisons.

```console
tesseract build solutions/tesseract
```

Use the Tesseract adapters for JAX or PyTorch when the compiled component must be
composed through those frameworks. Benchmarks distinguish an in-process Tesseract
call from container startup, serialization, and served requests.

The repository is organized around the common operation rather than around the
Tesseract schema. This keeps Tesseract comparable with direct Fortran, Python FFI,
and framework-native solutions.
