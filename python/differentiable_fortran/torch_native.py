"""PyTorch-native implementation of the common heat-step contract."""

from __future__ import annotations

import torch


def heat_step(x, alpha, dt, dx):
    """Advance the heat equation using only PyTorch tensor operations."""
    ratio = alpha * dt / dx**2
    interior = x[1:-1] + ratio * (x[:-2] - 2.0 * x[1:-1] + x[2:])
    return torch.cat((x[:1], interior, x[-1:]))


def jvp(x, alpha, dt, dx, x_dot, alpha_dot, dt_dot, dx_dot):
    """Return PyTorch's primal and forward-mode derivative."""
    return torch.func.jvp(
        heat_step,
        (x, alpha, dt, dx),
        (x_dot, alpha_dot, dt_dot, dx_dot),
    )


def vjp(x, alpha, dt, dx, y_bar):
    """Return PyTorch's reverse-mode derivative for all four inputs."""
    _, pullback = torch.func.vjp(heat_step, x, alpha, dt, dx)
    return pullback(y_bar)
