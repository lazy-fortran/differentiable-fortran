import numpy as np
import pytest

from differentiable_fortran import heat_step_jvp, heat_step_vjp


def inputs():
    grid = np.arange(19, dtype=np.float64)
    return (
        np.sin(grid * 0.3) + grid * 0.1,
        np.cos(grid * 0.2),
        np.where(grid.astype(int) % 2 == 0, 1.0, -1.0) / (grid + 2.0),
    )


def test_autograd_native_matches_analytical_derivatives() -> None:
    pytest.importorskip("autograd")
    from differentiable_fortran import autograd_native

    x, x_dot, y_bar = inputs()
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    actual_jvp = autograd_native.jvp(x, *parameters, x_dot, *parameter_dot)[1]
    expected_jvp = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2.0e-14)

    actual_vjp = autograd_native.vjp(x, *parameters, y_bar)
    expected_vjp = heat_step_vjp(x, *parameters, y_bar)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_jax_native_matches_analytical_derivatives() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from differentiable_fortran import jax_native

    jax.config.update("jax_enable_x64", True)
    x, x_dot, y_bar = inputs()
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    actual_jvp = jax_native.jvp(
        jnp.asarray(x),
        *(jnp.asarray(value) for value in parameters),
        jnp.asarray(x_dot),
        *(jnp.asarray(value) for value in parameter_dot),
    )[1]
    expected_jvp = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2.0e-14)

    actual_vjp = jax_native.vjp(
        jnp.asarray(x),
        *(jnp.asarray(value) for value in parameters),
        jnp.asarray(y_bar),
    )
    expected_vjp = heat_step_vjp(x, *parameters, y_bar)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_torch_native_matches_analytical_derivatives() -> None:
    torch = pytest.importorskip("torch")
    from differentiable_fortran import torch_native

    torch.set_default_dtype(torch.float64)
    x, x_dot, y_bar = inputs()
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    actual_jvp = torch_native.jvp(
        torch.from_numpy(x),
        *(torch.tensor(value) for value in parameters),
        torch.from_numpy(x_dot),
        *(torch.tensor(value) for value in parameter_dot),
    )[1]
    expected_jvp = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)[1]
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2.0e-14)

    actual_vjp = torch_native.vjp(
        torch.from_numpy(x),
        *(torch.tensor(value) for value in parameters),
        torch.from_numpy(y_bar),
    )
    expected_vjp = heat_step_vjp(x, *parameters, y_bar)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
