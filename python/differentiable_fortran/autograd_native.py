"""Eager NumPy-based automatic differentiation with HIPS Autograd."""

from __future__ import annotations

from autograd import make_jvp, make_vjp
import autograd.numpy as np


_ARGNUMS = (0, 1, 2, 3)


def heat_step(x, alpha, dt, dx):
    """Advance the heat equation with Autograd-compatible NumPy operations."""
    ratio = alpha * dt / dx**2
    interior = x[1:-1] + ratio * (x[:-2] - 2.0 * x[1:-1] + x[2:])
    return np.concatenate((x[:1], interior, x[-1:]))


def jvp(x, alpha, dt, dx, x_dot, alpha_dot, dt_dot, dx_dot):
    """Return the primal and forward-mode derivative without JIT compilation."""
    evaluate = make_jvp(heat_step, argnum=_ARGNUMS)(x, alpha, dt, dx)
    return evaluate((x_dot, alpha_dot, dt_dot, dx_dot))


def vjp(x, alpha, dt, dx, y_bar):
    """Return the reverse-mode derivative for all four inputs."""
    pullback, _ = make_vjp(heat_step, argnum=_ARGNUMS)(x, alpha, dt, dx)
    return pullback(y_bar)
