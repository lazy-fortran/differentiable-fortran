#include <stddef.h>

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;

extern void enzyme_study_objective(int workload, int n, int length, double *x,
                                   double *value);
extern void __enzyme_autodiff(void *, ...);

static int input_length(int workload, int n) {
  switch (workload) {
  case 2:
    return 3 * n;
  case 3:
    return 4 * n;
  case 6:
    return 2 * n;
  case 7:
    return 2 * n * n;
  default:
    return n;
  }
}

double enzyme_study_primal(int workload, int n, double *x) {
  double value;
  enzyme_study_objective(workload, n, input_length(workload, n), x, &value);
  return value;
}

double enzyme_study_vjp(int workload, int n, double *x, double *x_bar) {
  int length = input_length(workload, n);
  double value;
  double value_bar = 1.0;
  __enzyme_autodiff((void *)enzyme_study_objective, enzyme_const, workload,
                    enzyme_const, n, enzyme_const, length, enzyme_dup, x, x_bar,
                    enzyme_dupnoneed, &value, &value_bar);
  return x_bar[0];
}
