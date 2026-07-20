from pathlib import Path

import numpy as np
import pytest

from differentiable_fortran import NativeHeatStep, heat_step_jvp, heat_step_vjp


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("library", "enzyme"),
    [
        (ROOT / "build/libdifferentiable_fortran.so", False),
        (ROOT / "build/flang-enzyme/libdf_flang_enzyme.so", True),
        (ROOT / "build/lfortran-enzyme/libdf_lfortran_enzyme.so", True),
    ],
)
def test_native_derivatives_match_analytical_reference(
    library: Path, enzyme: bool
) -> None:
    if not library.exists():
        pytest.skip(f"optional library is not built: {library}")
    rng = np.random.default_rng(1729)
    x = rng.normal(size=31)
    x_dot = rng.normal(size=x.size)
    y_bar = rng.normal(size=x.size)
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    native = NativeHeatStep(library, enzyme=enzyme)

    actual_primal, actual_jvp = native.jvp(x, *parameters, x_dot, *parameter_dot)
    expected_primal, expected_jvp = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)
    np.testing.assert_allclose(actual_primal, expected_primal, rtol=2e-14)
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-13, atol=2e-14)

    actual_vjp = native.vjp(x, *parameters, y_bar)
    expected_vjp = heat_step_vjp(x, *parameters, y_bar)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
