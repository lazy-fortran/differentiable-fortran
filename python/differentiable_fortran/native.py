"""Typed ctypes access to analytical and Enzyme shared libraries."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_ARRAY = ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED"))
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def default_library() -> Path:
    """Return the analytical library built by the repository CMake project."""
    override = os.environ.get("DF_LIBRARY")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "build/libdifferentiable_fortran.so"


class NativeHeatStep:
    """A checked, coarse-grained ctypes wrapper around the common C ABI."""

    def __init__(self, library: str | Path | None = None, *, enzyme: bool = False):
        path = Path(library) if library is not None else default_library()
        self.path = path.resolve()
        self.library = ctypes.CDLL(str(self.path))
        prefix = "df_enzyme_heat_step" if enzyme else "df_heat_step"
        self._primal = getattr(self.library, prefix)
        self._jvp = getattr(self.library, f"{prefix}_jvp")
        self._vjp = getattr(self.library, f"{prefix}_vjp")
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        scalar_inputs = [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._primal.argtypes = [*scalar_inputs, _ARRAY, _ARRAY]
        self._primal.restype = None
        self._jvp.argtypes = [
            *scalar_inputs,
            _ARRAY,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            _ARRAY,
            _ARRAY,
            _ARRAY,
        ]
        self._jvp.restype = None
        self._vjp.argtypes = [
            *scalar_inputs,
            _ARRAY,
            _ARRAY,
            _ARRAY,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
        ]
        self._vjp.restype = None

    @staticmethod
    def _array(value: ArrayLike) -> FloatArray:
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.ndim != 1 or result.size < 2:
            raise ValueError("arrays must be one-dimensional with length >= 2")
        return result

    def primal(self, x: ArrayLike, alpha: float, dt: float, dx: float) -> FloatArray:
        state = self._array(x)
        result = np.empty_like(state)
        self._primal(state.size, alpha, dt, dx, state, result)
        return result

    def jvp(
        self,
        x: ArrayLike,
        alpha: float,
        dt: float,
        dx: float,
        x_dot: ArrayLike,
        alpha_dot: float,
        dt_dot: float,
        dx_dot: float,
    ) -> tuple[FloatArray, FloatArray]:
        state = self._array(x)
        tangent = self._array(x_dot)
        if state.shape != tangent.shape:
            raise ValueError("x and x_dot must have the same shape")
        result = np.empty_like(state)
        result_dot = np.empty_like(state)
        self._jvp(
            state.size,
            alpha,
            dt,
            dx,
            state,
            alpha_dot,
            dt_dot,
            dx_dot,
            tangent,
            result,
            result_dot,
        )
        return result, result_dot

    def vjp(
        self,
        x: ArrayLike,
        alpha: float,
        dt: float,
        dx: float,
        y_bar: ArrayLike,
    ) -> tuple[FloatArray, float, float, float]:
        state = self._array(x)
        cotangent = self._array(y_bar)
        if state.shape != cotangent.shape:
            raise ValueError("x and y_bar must have the same shape")
        x_bar = np.zeros_like(state)
        alpha_bar = ctypes.c_double()
        dt_bar = ctypes.c_double()
        dx_bar = ctypes.c_double()
        self._vjp(
            state.size,
            alpha,
            dt,
            dx,
            state,
            cotangent,
            x_bar,
            ctypes.byref(alpha_bar),
            ctypes.byref(dt_bar),
            ctypes.byref(dx_bar),
        )
        return x_bar, alpha_bar.value, dt_bar.value, dx_bar.value
