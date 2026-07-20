from __future__ import annotations

import torch

# Importing the extension registers both operators with the dispatcher.
import _df_torch_operator  # noqa: F401


@torch.library.register_fake("differentiable_fortran::heat_step")
def _heat_step_fake(x, alpha, dt, dx):
    return torch.empty_like(x)


@torch.library.register_fake("differentiable_fortran::heat_step_vjp")
def _heat_step_vjp_fake(x, alpha, dt, dx, y_bar):
    return (
        torch.empty_like(x),
        torch.empty_like(alpha),
        torch.empty_like(dt),
        torch.empty_like(dx),
    )


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, y_bar):
    x, alpha, dt, dx = ctx.saved_tensors
    return torch.ops.differentiable_fortran.heat_step_vjp(
        x, alpha, dt, dx, y_bar.contiguous()
    )


torch.library.register_autograd(
    "differentiable_fortran::heat_step",
    _backward,
    setup_context=_setup_context,
)


def heat_step(x, alpha, dt, dx):
    return torch.ops.differentiable_fortran.heat_step(x, alpha, dt, dx)
