# Flang plus Enzyme

This solution compiles the minimal `bind(C)` Fortran kernel to LLVM IR with
Flang, adds stable JVP and VJP entry points in a C wrapper, runs Enzyme over the
linked IR, and produces `libdf_flang_enzyme.so`.

Build the ordinary analytical library first, then the generated derivative:

```console
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
toolchains/flang-enzyme/build.sh
```

The smoke program compares the Enzyme JVP and VJP with the analytical Fortran
implementations, including derivatives with respect to the state, diffusivity,
time step, and grid spacing.
