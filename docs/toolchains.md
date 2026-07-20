# Compiler and Enzyme toolchains

Enzyme transforms LLVM IR, so the compiler, `opt`, and Enzyme plugin must use
compatible LLVM versions. A compiler executable having the right version number is
not sufficient: inspect emitted IR and load the plugin with that version's `opt`
before comparing runtime results.

The Flang pipeline emits LLVM IR directly, runs Enzyme's transformation pass, and
links the transformed module with a small C ABI wrapper. The LFortran pipeline
performs the same stages with LFortran-generated LLVM IR. Both consume the same
minimal primal source and export the same operations.

Each build script records commands and version output in its build directory. Use
`LLVM_MAJOR`, `ENZYME_PLUGIN`, and compiler-specific environment variables to select
non-default installations. The scripts stop on a version mismatch rather than
silently mixing toolchains.

The ordinary CMake build does not require Enzyme. This lets the analytical oracle,
ISO C library, object-oriented API, and Python reference tests run on machines that
do not have an Enzyme installation.
