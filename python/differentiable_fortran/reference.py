"""NumPy analytical reference for the common heat-step contract."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _state(value: ArrayLike) -> FloatArray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError("state arrays must be one-dimensional with length >= 2")
    return result


def heat_step(x: ArrayLike, alpha: float, dt: float, dx: float) -> FloatArray:
    """Advance the one-dimensional heat equation by one explicit step."""
    state = _state(x)
    ratio = alpha * dt / dx**2
    result = state.copy()
    result[1:-1] += ratio * (state[:-2] - 2.0 * state[1:-1] + state[2:])
    return result


def heat_step_jvp(
    x: ArrayLike,
    alpha: float,
    dt: float,
    dx: float,
    x_dot: ArrayLike,
    alpha_dot: float,
    dt_dot: float,
    dx_dot: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the primal result and its analytical Jacobian-vector product."""
    state = _state(x)
    state_dot = _state(x_dot)
    if state.shape != state_dot.shape:
        raise ValueError("x and x_dot must have the same shape")
    ratio = alpha * dt / dx**2
    ratio_dot = (
        dt * alpha_dot / dx**2
        + alpha * dt_dot / dx**2
        - 2.0 * alpha * dt * dx_dot / dx**3
    )
    laplacian = state[:-2] - 2.0 * state[1:-1] + state[2:]
    tangent_laplacian = state_dot[:-2] - 2.0 * state_dot[1:-1] + state_dot[2:]
    result = heat_step(state, alpha, dt, dx)
    result_dot = state_dot.copy()
    result_dot[1:-1] += ratio_dot * laplacian + ratio * tangent_laplacian
    return result, result_dot


def heat_step_vjp(
    x: ArrayLike,
    alpha: float,
    dt: float,
    dx: float,
    y_bar: ArrayLike,
) -> tuple[FloatArray, float, float, float]:
    """Return the analytical vector-Jacobian product for all inputs."""
    state = _state(x)
    cotangent = _state(y_bar)
    if state.shape != cotangent.shape:
        raise ValueError("x and y_bar must have the same shape")
    ratio = alpha * dt / dx**2
    x_bar = np.zeros_like(state)
    x_bar[0] = cotangent[0]
    x_bar[-1] = cotangent[-1]
    interior = cotangent[1:-1]
    x_bar[:-2] += ratio * interior
    x_bar[1:-1] += (1.0 - 2.0 * ratio) * interior
    x_bar[2:] += ratio * interior
    laplacian = state[:-2] - 2.0 * state[1:-1] + state[2:]
    laplacian_sum = float(np.dot(interior, laplacian))
    return (
        x_bar,
        laplacian_sum * dt / dx**2,
        laplacian_sum * alpha / dx**2,
        -2.0 * laplacian_sum * alpha * dt / dx**3,
    )
