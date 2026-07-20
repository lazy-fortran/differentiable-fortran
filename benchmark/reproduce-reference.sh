#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

uv sync \
    --extra benchmark \
    --extra jax \
    --extra tesseract \
    --extra torch \
    --extra wrappers

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
toolchains/flang-enzyme/build.sh
toolchains/lfortran-enzyme/build.sh
solutions/f90wrap/build.sh
solutions/cython/build.sh
solutions/numpy-ufunc/build.sh
solutions/jax-ffi/build.sh
solutions/pytorch-custom-op/build.sh
uv run benchmark/run.py
uv run benchmark/plot.py
