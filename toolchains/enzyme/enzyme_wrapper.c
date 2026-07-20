#include <stddef.h>

extern int enzyme_const;
extern int enzyme_dup;

extern void df_enzyme_heat_step(int n, double alpha, double dt, double dx,
                                const double *x, double *y);
extern double df_enzyme_heat_objective(int n, double alpha, double dt,
                                       double dx, const double *x,
                                       const double *y_bar);

extern void __enzyme_fwddiff(void *, ...);

typedef struct {
  double alpha;
  double dt;
  double dx;
} scalar_gradients_t;

extern scalar_gradients_t __enzyme_autodiff(void *, ...);

void df_enzyme_heat_step_jvp(int n, double alpha, double dt, double dx,
                             const double *x, double alpha_dot, double dt_dot,
                             double dx_dot, const double *x_dot, double *y,
                             double *y_dot) {
  __enzyme_fwddiff((void *)df_enzyme_heat_step, enzyme_const, n, alpha,
                   alpha_dot, dt, dt_dot, dx, dx_dot, enzyme_dup, x, x_dot,
                   enzyme_dup, y, y_dot);
}

void df_enzyme_heat_step_vjp(int n, double alpha, double dt, double dx,
                             const double *x, const double *y_bar,
                             double *x_bar, double *alpha_bar, double *dt_bar,
                             double *dx_bar) {
  scalar_gradients_t gradients = __enzyme_autodiff(
      (void *)df_enzyme_heat_objective, enzyme_const, n, alpha, dt, dx,
      enzyme_dup, x, x_bar, enzyme_const, y_bar);

  *alpha_bar = gradients.alpha;
  *dt_bar = gradients.dt;
  *dx_bar = gradients.dx;
}
