#ifndef DIFFERENTIABLE_FORTRAN_HEAT_STEP_H
#define DIFFERENTIABLE_FORTRAN_HEAT_STEP_H

#ifdef __cplusplus
extern "C" {
#endif

void df_heat_step(int n, double alpha, double dt, double dx, const double *x,
                  double *y);

void df_heat_step_jvp(int n, double alpha, double dt, double dx,
                      const double *x, double alpha_dot, double dt_dot,
                      double dx_dot, const double *x_dot, double *y,
                      double *y_dot);

void df_heat_step_vjp(int n, double alpha, double dt, double dx,
                      const double *x, const double *y_bar, double *x_bar,
                      double *alpha_bar, double *dt_bar, double *dx_bar);

double df_gaussian(double x);
double df_heat_point(double left, double center, double right, double ratio);

#ifdef __cplusplus
}
#endif

#endif
