#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${BUILD_DIR:-"$root/build/jax-ffi"}
include_dir=$("$root/.venv/bin/python" -c 'import jax; print(jax.ffi.include_dir())')

mkdir -p "$build_dir"
c++ -std=c++17 -O3 -fPIC -shared \
    -I"$include_dir" \
    "$root/solutions/jax-ffi/handler.cc" \
    -L"$root/build" -ldifferentiable_fortran \
    -Wl,-rpath,"$root/build" \
    -o "$build_dir/libdf_jax_ffi.so"

PYTHONPATH="$root/solutions/jax-ffi" "$root/.venv/bin/python" \
    "$root/solutions/jax-ffi/smoke.py"
