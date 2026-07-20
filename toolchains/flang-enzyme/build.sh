#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${BUILD_DIR:-"$root/build/flang-enzyme"}
flang=${FLANG:-flang-new}
clang=${CLANG:-clang-22}
opt=${OPT:-opt}
llvm_link=${LLVM_LINK:-llvm-link}
plugin=${ENZYME_PLUGIN:?Set ENZYME_PLUGIN to the LLVMEnzyme shared library}

mkdir -p "$build_dir"

"$flang" --version > "$build_dir/versions.txt"
"$clang" --version >> "$build_dir/versions.txt"
"$opt" --version >> "$build_dir/versions.txt"
test -f "$plugin"

"$flang" -O3 -fPIC -S -emit-llvm \
    "$root/toolchains/enzyme/heat_step_kernel.f90" \
    -o "$build_dir/kernel.ll"
"$clang" -O3 -fPIC -S -emit-llvm \
    "$root/toolchains/enzyme/enzyme_wrapper.c" \
    -o "$build_dir/wrapper.ll"
"$llvm_link" -S "$build_dir/kernel.ll" "$build_dir/wrapper.ll" \
    -o "$build_dir/linked.ll"
"$opt" -load-pass-plugin="$plugin" -passes=enzyme \
    "$build_dir/linked.ll" -S -o "$build_dir/enzyme.ll"
"$flang" -O3 -fPIC -shared "$build_dir/enzyme.ll" \
    -o "$build_dir/libdf_flang_enzyme.so"

"$clang" -O2 "$root/toolchains/enzyme/smoke.c" \
    -L"$build_dir" -ldf_flang_enzyme \
    -L"$root/build" -ldifferentiable_fortran \
    -Wl,-rpath,"$build_dir" -Wl,-rpath,"$root/build" \
    -o "$build_dir/smoke"
"$build_dir/smoke"
