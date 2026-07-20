from __future__ import annotations

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ctypes.CDLL(str(ROOT / "build/jax-ffi/libdf_jax_ffi.so"))

jax.ffi.register_ffi_target(
    "df_heat_step", jax.ffi.pycapsule(LIBRARY.DfHeatStep), platform="cpu"
)
jax.ffi.register_ffi_target(
    "df_heat_step_jvp", jax.ffi.pycapsule(LIBRARY.DfHeatStepJvp), platform="cpu"
)
jax.ffi.register_ffi_target(
    "df_heat_step_vjp", jax.ffi.pycapsule(LIBRARY.DfHeatStepVjp), platform="cpu"
)


def _scalar(value):
    return jnp.asarray(value, dtype=jnp.float64)


def primal(x, alpha, dt, dx):
    result = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    return jax.ffi.ffi_call("df_heat_step", result)(
        x, _scalar(alpha), _scalar(dt), _scalar(dx)
    )


def jvp(x, alpha, dt, dx, x_dot, alpha_dot, dt_dot, dx_dot):
    result = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    return jax.ffi.ffi_call("df_heat_step_jvp", (result, result))(
        x,
        _scalar(alpha),
        _scalar(dt),
        _scalar(dx),
        x_dot,
        _scalar(alpha_dot),
        _scalar(dt_dot),
        _scalar(dx_dot),
    )


def vjp(x, alpha, dt, dx, y_bar):
    vector = jax.ShapeDtypeStruct(x.shape, jnp.float64)
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.ffi.ffi_call("df_heat_step_vjp", (vector, scalar, scalar, scalar))(
        x, _scalar(alpha), _scalar(dt), _scalar(dx), y_bar
    )
