#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

extern "C" {
void df_heat_step(int n, double alpha, double dt, double dx, const double *x,
                  double *y);
void df_heat_step_jvp(int n, double alpha, double dt, double dx,
                      const double *x, double alpha_dot, double dt_dot,
                      double dx_dot, const double *x_dot, double *y,
                      double *y_dot);
void df_heat_step_vjp(int n, double alpha, double dt, double dx,
                      const double *x, const double *y_bar, double *x_bar,
                      double *alpha_bar, double *dt_bar, double *dx_bar);
}

namespace ffi = xla::ffi;

using Buffer = ffi::Buffer<ffi::F64>;
using ResultBuffer = ffi::ResultBuffer<ffi::F64>;

ffi::Error HeatStepImpl(Buffer x, Buffer alpha, Buffer dt, Buffer dx,
                        ResultBuffer y) {
  if (x.dimensions().size() != 1 || x.element_count() < 2) {
    return ffi::Error::InvalidArgument("x must be a vector with length >= 2");
  }
  df_heat_step(static_cast<int>(x.element_count()), alpha.typed_data()[0],
               dt.typed_data()[0], dx.typed_data()[0], x.typed_data(),
               y->typed_data());
  return ffi::Error::Success();
}

ffi::Error HeatStepJvpImpl(Buffer x, Buffer alpha, Buffer dt, Buffer dx,
                           Buffer x_dot, Buffer alpha_dot, Buffer dt_dot,
                           Buffer dx_dot, ResultBuffer y, ResultBuffer y_dot) {
  df_heat_step_jvp(static_cast<int>(x.element_count()), alpha.typed_data()[0],
                   dt.typed_data()[0], dx.typed_data()[0], x.typed_data(),
                   alpha_dot.typed_data()[0], dt_dot.typed_data()[0],
                   dx_dot.typed_data()[0], x_dot.typed_data(), y->typed_data(),
                   y_dot->typed_data());
  return ffi::Error::Success();
}

ffi::Error HeatStepVjpImpl(Buffer x, Buffer alpha, Buffer dt, Buffer dx,
                           Buffer y_bar, ResultBuffer x_bar,
                           ResultBuffer alpha_bar, ResultBuffer dt_bar,
                           ResultBuffer dx_bar) {
  df_heat_step_vjp(static_cast<int>(x.element_count()), alpha.typed_data()[0],
                   dt.typed_data()[0], dx.typed_data()[0], x.typed_data(),
                   y_bar.typed_data(), x_bar->typed_data(),
                   alpha_bar->typed_data(), dt_bar->typed_data(),
                   dx_bar->typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DfHeatStep, HeatStepImpl,
                              ffi::Ffi::Bind()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Ret<Buffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DfHeatStepJvp, HeatStepJvpImpl,
                              ffi::Ffi::Bind()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Ret<Buffer>()
                                  .Ret<Buffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(DfHeatStepVjp, HeatStepVjpImpl,
                              ffi::Ffi::Bind()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Arg<Buffer>()
                                  .Ret<Buffer>()
                                  .Ret<Buffer>()
                                  .Ret<Buffer>()
                                  .Ret<Buffer>());
