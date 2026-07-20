#include <torch/extension.h>

#include <tuple>

extern "C" {
void df_heat_step(int n, double alpha, double dt, double dx, const double *x,
                  double *y);
void df_heat_step_vjp(int n, double alpha, double dt, double dx,
                      const double *x, const double *y_bar, double *x_bar,
                      double *alpha_bar, double *dt_bar, double *dx_bar);
}

namespace {

void CheckTensor(const at::Tensor &value, const char *name) {
  TORCH_CHECK(value.device().is_cpu(), name, " must be on the CPU");
  TORCH_CHECK(value.scalar_type() == at::kDouble, name, " must be float64");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

at::Tensor HeatStep(const at::Tensor &x, const at::Tensor &alpha,
                    const at::Tensor &dt, const at::Tensor &dx) {
  CheckTensor(x, "x");
  CheckTensor(alpha, "alpha");
  CheckTensor(dt, "dt");
  CheckTensor(dx, "dx");
  TORCH_CHECK(x.dim() == 1 && x.numel() >= 2,
              "x must be a vector with length >= 2");
  at::Tensor y = at::empty_like(x);
  df_heat_step(static_cast<int>(x.numel()), alpha.item<double>(),
               dt.item<double>(), dx.item<double>(), x.data_ptr<double>(),
               y.data_ptr<double>());
  return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
HeatStepVjp(const at::Tensor &x, const at::Tensor &alpha, const at::Tensor &dt,
            const at::Tensor &dx, const at::Tensor &y_bar) {
  CheckTensor(x, "x");
  CheckTensor(y_bar, "y_bar");
  at::Tensor x_bar = at::zeros_like(x);
  at::Tensor alpha_bar = at::empty_like(alpha);
  at::Tensor dt_bar = at::empty_like(dt);
  at::Tensor dx_bar = at::empty_like(dx);
  df_heat_step_vjp(static_cast<int>(x.numel()), alpha.item<double>(),
                   dt.item<double>(), dx.item<double>(), x.data_ptr<double>(),
                   y_bar.data_ptr<double>(), x_bar.data_ptr<double>(),
                   alpha_bar.data_ptr<double>(), dt_bar.data_ptr<double>(),
                   dx_bar.data_ptr<double>());
  return std::make_tuple(x_bar, alpha_bar, dt_bar, dx_bar);
}

} // namespace

TORCH_LIBRARY(differentiable_fortran, library) {
  library.def(
      "heat_step(Tensor x, Tensor alpha, Tensor dt, Tensor dx) -> Tensor");
  library.def("heat_step_vjp(Tensor x, Tensor alpha, Tensor dt, Tensor dx, "
              "Tensor y_bar) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(differentiable_fortran, CPU, library) {
  library.impl("heat_step", HeatStep);
  library.impl("heat_step_vjp", HeatStepVjp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
