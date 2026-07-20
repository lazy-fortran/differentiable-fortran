#!/usr/bin/env bash
set -euo pipefail

study=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$study/../.." && pwd)

"$study/build.sh"
OMP_NUM_THREADS=1 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    "$root/.venv/bin/python" "$study/run.py"
"$root/.venv/bin/python" "$study/plot.py"
