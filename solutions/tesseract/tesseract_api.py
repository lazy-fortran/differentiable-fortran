"""Tesseract API for the common heat-step contract and Enzyme derivatives."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


_library_path = Path(
    os.environ.get("DF_TESSERACT_LIBRARY", "/tesseract/lib/libdf_lfortran_enzyme.so")
)
_library = ctypes.CDLL(str(_library_path))
_prefix = "df_enzyme_heat_step"
_array_pointer = ctypes.POINTER(ctypes.c_double)

_primal = getattr(_library, _prefix)
_primal.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    _array_pointer,
    _array_pointer,
]
_primal.restype = None

_jvp = getattr(_library, f"{_prefix}_jvp")
_jvp.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    _array_pointer,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    _array_pointer,
    _array_pointer,
    _array_pointer,
]
_jvp.restype = None

_vjp = getattr(_library, f"{_prefix}_vjp")
_vjp.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    _array_pointer,
    _array_pointer,
    _array_pointer,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_vjp.restype = None


def _array(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def _pointer(value: np.ndarray) -> _array_pointer:
    return value.ctypes.data_as(_array_pointer)


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]] = Field(
        description="State vector before the heat step"
    )
    alpha: Differentiable[Float64] = Field(description="Thermal diffusivity")
    dt: Differentiable[Float64] = Field(description="Time-step size")
    dx: Differentiable[Float64] = Field(description="Grid spacing")

    @model_validator(mode="after")
    def validate_contract(self):
        if isinstance(self.x, ShapeDType):
            return self
        if len(self.x) < 2:
            raise ValueError("x must have length >= 2")
        if self.dx == 0.0:
            raise ValueError("dx must be nonzero")
        return self


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(
        description="State vector after the heat step"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    x = _array(inputs.x)
    y = np.empty_like(x)
    _primal(len(x), inputs.alpha, inputs.dt, inputs.dx, _pointer(x), _pointer(y))
    return OutputSchema(y=y)


def abstract_eval(abstract_inputs):
    return {
        "y": ShapeDType(shape=abstract_inputs.x.shape, dtype=abstract_inputs.x.dtype)
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    x = _array(inputs.x)
    x_dot = _array(tangent_vector.get("x", np.zeros_like(x)))
    y = np.empty_like(x)
    y_dot = np.empty_like(x)
    _jvp(
        len(x),
        inputs.alpha,
        inputs.dt,
        inputs.dx,
        _pointer(x),
        tangent_vector.get("alpha", 0.0),
        tangent_vector.get("dt", 0.0),
        tangent_vector.get("dx", 0.0),
        _pointer(x_dot),
        _pointer(y),
        _pointer(y_dot),
    )
    return {"y": y_dot} if "y" in jvp_outputs else {}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    x = _array(inputs.x)
    y_bar = _array(cotangent_vector.get("y", np.zeros_like(x)))
    x_bar = np.zeros_like(x)
    alpha_bar = ctypes.c_double()
    dt_bar = ctypes.c_double()
    dx_bar = ctypes.c_double()
    _vjp(
        len(x),
        inputs.alpha,
        inputs.dt,
        inputs.dx,
        _pointer(x),
        _pointer(y_bar),
        _pointer(x_bar),
        ctypes.byref(alpha_bar),
        ctypes.byref(dt_bar),
        ctypes.byref(dx_bar),
    )
    all_results = {
        "x": x_bar,
        "alpha": alpha_bar.value,
        "dt": dt_bar.value,
        "dx": dx_bar.value,
    }
    return {name: all_results[name] for name in vjp_inputs}
