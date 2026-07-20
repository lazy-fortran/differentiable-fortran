import numpy as np

from differentiable_fortran import heat_step, heat_step_jvp, heat_step_vjp


def test_jvp_matches_centered_finite_difference() -> None:
    x = np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1
    x_dot = np.cos(np.arange(12) * 0.2)
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    epsilon = 1.0e-7
    _, actual = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)
    plus = heat_step(
        x + epsilon * x_dot,
        *(np.asarray(parameters) + epsilon * np.asarray(parameter_dot)),
    )
    minus = heat_step(
        x - epsilon * x_dot,
        *(np.asarray(parameters) - epsilon * np.asarray(parameter_dot)),
    )
    np.testing.assert_allclose(actual, (plus - minus) / (2.0 * epsilon), rtol=2e-8)


def test_jvp_and_vjp_satisfy_adjoint_identity() -> None:
    rng = np.random.default_rng(1729)
    x = rng.normal(size=17)
    x_dot = rng.normal(size=x.size)
    y_bar = rng.normal(size=x.size)
    parameters = (0.7, 0.02, 0.2)
    parameter_dot = (0.13, -0.01, 0.03)
    _, y_dot = heat_step_jvp(x, *parameters, x_dot, *parameter_dot)
    x_bar, alpha_bar, dt_bar, dx_bar = heat_step_vjp(x, *parameters, y_bar)
    left = np.dot(y_dot, y_bar)
    right = np.dot(x_dot, x_bar) + np.dot(parameter_dot, (alpha_bar, dt_bar, dx_bar))
    np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)
