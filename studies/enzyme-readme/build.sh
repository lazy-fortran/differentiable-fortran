#!/usr/bin/env bash
set -euo pipefail

study=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$study/../.." && pwd)
build=${BUILD_DIR:-"$root/build/enzyme-readme"}
plugin=${ENZYME_PLUGIN:?Set ENZYME_PLUGIN to the LLVMEnzyme shared library}
clang=${CLANG:-clang-22}
flang=${FLANG:-flang-new}
lfortran=${LFORTRAN:-lfortran}
opt=${OPT:-opt}
llvm_link=${LLVM_LINK:-llvm-link}

mkdir -p "$build"
test -f "$plugin"

"$clang" -O3 -fPIC -std=c++17 -S -emit-llvm "$study/source/kernels.cpp" \
    -o "$build/cpp.ll"
"$opt" -load-pass-plugin="$plugin" -passes=enzyme "$build/cpp.ll" \
    -S -o "$build/cpp-enzyme.ll"
"$clang" -O3 -fPIC -shared "$build/cpp-enzyme.ll" -lstdc++ \
    -o "$build/libstudy_cpp_enzyme.so"

"$clang" -O3 -fPIC -S -emit-llvm "$study/source/enzyme_wrapper.c" \
    -o "$build/wrapper.ll"

"$flang" -O3 -fPIC -S -emit-llvm "$study/source/kernels.f90" \
    -o "$build/flang-kernels.ll"
"$llvm_link" -S "$build/flang-kernels.ll" "$build/wrapper.ll" \
    -o "$build/flang-linked.ll"
"$opt" -load-pass-plugin="$plugin" -passes=enzyme "$build/flang-linked.ll" \
    -S -o "$build/flang-enzyme.ll"
"$flang" -O3 -fPIC -shared "$build/flang-enzyme.ll" \
    -o "$build/libstudy_flang_enzyme.so"

"$lfortran" --show-llvm --fast "$study/source/kernels.f90" \
    > "$build/lfortran-kernels.ll"
# LFortran's runtime wrappers hide these elementary operations from Enzyme.
# Retarget their declarations to the corresponding C math symbols in generated IR.
sed -i \
    -e 's/@_lfortran_dlog/@log/g' \
    -e 's/@_lfortran_dsin/@sin/g' \
    -e 's/@_lfortran_dcos/@cos/g' \
    -e 's/@_lfortran_dtanh/@tanh/g' \
    "$build/lfortran-kernels.ll"
"$llvm_link" -S "$build/lfortran-kernels.ll" "$build/wrapper.ll" \
    -o "$build/lfortran-linked.ll"
"$opt" -load-pass-plugin="$plugin" -passes=enzyme \
    "$build/lfortran-linked.ll" -S -o "$build/lfortran-enzyme.ll"
"$clang" -O3 -fPIC -shared "$build/lfortran-enzyme.ll" \
    -o "$build/libstudy_lfortran_enzyme.so"

{
    "$clang" --version | sed -n '1p'
    "$flang" --version | sed -n '1p'
    "$lfortran" --version | sed -n '1p'
    "$opt" --version | sed -n '2s/^ *//p'
    sha256sum "$plugin" | awk '{print "Enzyme plugin SHA-256 " $1}'
} > "$build/versions.txt"
