# LFortran plus Enzyme

This solution compiles the same minimal `bind(C)` Fortran kernel to LLVM IR with
LFortran. The wrapper, Enzyme pass, public operations, analytical oracle, and smoke
test are identical to the Flang solution, so compiler effects can be measured
without changing the model.

```console
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
toolchains/lfortran-enzyme/build.sh
```

Set `LFORTRAN`, `OPT`, `CLANG`, `LLVM_LINK`, or `ENZYME_PLUGIN` to select a
different matching LLVM toolchain.
