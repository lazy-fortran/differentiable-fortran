#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root/solutions/cython"
"$root/.venv/bin/python" setup.py build_ext --inplace
PYTHONPATH=. "$root/.venv/bin/python" smoke.py
