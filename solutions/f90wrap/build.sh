#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${BUILD_DIR:-"$root/build/f90wrap"}

rm -rf "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"

# Keep the Fortran runtime as an explicit dependency of the extension even
# when the platform linker enables --as-needed by default.
export LDFLAGS="${LDFLAGS:--shared} -Wl,--no-as-needed -lgfortran"

"$root/.venv/bin/f90wrap" \
    -k "$root/solutions/f90wrap/kind_map" \
    -C new_heat_model \
    --only new_heat_model get_alpha get_dx set_alpha set_dx step step_jvp \
    -m heat_model \
    "$root/src/fortran/heat_step_primal.f90" \
    "$root/src/fortran/heat_step_analytic.f90" \
    "$root/src/fortran/heat_model.f90"

"$root/.venv/bin/f90wrap" --build \
    -k "$root/solutions/f90wrap/kind_map" \
    --only new_heat_model get_alpha get_dx set_alpha set_dx step step_jvp \
    -m heat_model \
    "$root/src/fortran/heat_step_primal.f90" \
    "$root/src/fortran/heat_step_analytic.f90" \
    "$root/src/fortran/heat_model.f90"

PYTHONPATH="$build_dir" "$root/.venv/bin/python" \
    "$root/solutions/f90wrap/smoke.py"
