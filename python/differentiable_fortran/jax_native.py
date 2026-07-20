"""JAX-native implementation of the common heat-step contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def heat_step(x, alpha, dt, dx):
    """Advance the heat equation using only JAX primitives."""
    ratio = alpha * dt / dx**2
    interior = x[1:-1] + ratio * (x[:-2] - 2.0 * x[1:-1] + x[2:])
    return jnp.concatenate((x[:1], interior, x[-1:]))


def jvp(x, alpha, dt, dx, x_dot, alpha_dot, dt_dot, dx_dot):
    """Return JAX's primal and forward-mode derivative."""
    return jax.jvp(
        heat_step,
        (x, alpha, dt, dx),
        (x_dot, alpha_dot, dt_dot, dx_dot),
    )


def vjp(x, alpha, dt, dx, y_bar):
    """Return JAX's reverse-mode derivative for all four inputs."""
    _, pullback = jax.vjp(heat_step, x, alpha, dt, dx)
    return pullback(y_bar)
