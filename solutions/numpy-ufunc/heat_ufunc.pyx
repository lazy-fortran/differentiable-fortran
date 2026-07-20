# cython: language_level=3

cimport cython


cdef extern from "heat_step.h":
    double df_heat_point(double left, double center, double right,
                         double ratio) noexcept nogil


@cython.ufunc
cdef double heat_interior(double left, double center, double right, double ratio):
    return df_heat_point(left, center, right, ratio)
