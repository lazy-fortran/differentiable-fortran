# cython: language_level=3

import numpy as np
cimport numpy as cnp


cdef extern from "heat_step.h":
    void df_heat_step(int n, double alpha, double dt, double dx,
                      const double *x, double *y) noexcept nogil


def primal(const double[::1] x, double alpha, double dt, double dx):
    """Call the Fortran heat step through a compiled Cython adapter."""
    if x.shape[0] < 2:
        raise ValueError("x must have length >= 2")
    cdef cnp.ndarray[cnp.double_t, ndim=1] y = np.empty(x.shape[0], dtype=np.float64)
    cdef double[::1] y_view = y
    with nogil:
        df_heat_step(x.shape[0], alpha, dt, dx, &x[0], &y_view[0])
    return y
