#include "../../contract/heat_step.h"

#include <math.h>
#include <stdio.h>

extern void df_enzyme_heat_step(int n, double alpha, double dt, double dx,
                                const double *x, double *y);
extern void df_enzyme_heat_step_jvp(int n, double alpha, double dt, double dx,
                                    const double *x, double alpha_dot,
                                    double dt_dot, double dx_dot,
                                    const double *x_dot, double *y,
                                    double *y_dot);
extern void df_enzyme_heat_step_vjp(int n, double alpha, double dt, double dx,
                                    const double *x, const double *y_bar,
                                    double *x_bar, double *alpha_bar,
                                    double *dt_bar, double *dx_bar);

static int close_enough(double actual, double expected) {
  double scale = 1.0 + fabs(expected);
  return fabs(actual - expected) <= 5.0e-12 * scale;
}

int main(void) {
  enum { n = 8 };
  double x[n], x_dot[n], y[n], y_dot[n], y_expected[n], y_dot_expected[n];
  double y_bar[n], x_bar[n], x_bar_expected[n];
  double alpha_bar, dt_bar, dx_bar;
  double alpha_bar_expected, dt_bar_expected, dx_bar_expected;
  int i;

  for (i = 0; i < n; ++i) {
    x[i] = sin(0.3 * i) + 0.1 * i;
    x_dot[i] = cos(0.2 * i);
    y_bar[i] = (i % 2 == 0 ? 1.0 : -1.0) / (i + 2.0);
    x_bar[i] = 0.0;
  }

  df_enzyme_heat_step_jvp(n, 0.7, 0.02, 0.2, x, 0.13, -0.01, 0.03, x_dot, y,
                          y_dot);
  df_heat_step_jvp(n, 0.7, 0.02, 0.2, x, 0.13, -0.01, 0.03, x_dot, y_expected,
                   y_dot_expected);
  for (i = 0; i < n; ++i) {
    if (!close_enough(y[i], y_expected[i]) ||
        !close_enough(y_dot[i], y_dot_expected[i])) {
      fprintf(stderr, "JVP mismatch at %d: %.17g %.17g\n", i, y_dot[i],
              y_dot_expected[i]);
      return 1;
    }
  }

  df_enzyme_heat_step_vjp(n, 0.7, 0.02, 0.2, x, y_bar, x_bar, &alpha_bar,
                          &dt_bar, &dx_bar);
  df_heat_step_vjp(n, 0.7, 0.02, 0.2, x, y_bar, x_bar_expected,
                   &alpha_bar_expected, &dt_bar_expected, &dx_bar_expected);
  for (i = 0; i < n; ++i) {
    if (!close_enough(x_bar[i], x_bar_expected[i])) {
      fprintf(stderr, "VJP mismatch at %d: %.17g %.17g\n", i, x_bar[i],
              x_bar_expected[i]);
      return 1;
    }
  }
  if (!close_enough(alpha_bar, alpha_bar_expected) ||
      !close_enough(dt_bar, dt_bar_expected) ||
      !close_enough(dx_bar, dx_bar_expected)) {
    fprintf(stderr, "scalar VJP mismatch\n");
    return 1;
  }

  puts("PASS");
  return 0;
}
