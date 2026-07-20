#include <algorithm>
#include <cmath>
#include <cstddef>

extern "C" {
extern int enzyme_const;
extern int enzyme_dup;
extern double __enzyme_autodiff(void *, ...);
}

namespace {

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double lstm(int n, const double *x) {
  double cell = 0.2;
  double hidden = -0.1;
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    const double forget = sigmoid(0.7 * x[i] + 0.2);
    const double ingate = sigmoid(-0.4 * hidden + 0.1);
    const double outgate = sigmoid(0.5 * x[i] - 0.3);
    const double change = std::tanh(0.8 * hidden + 0.6 * x[i]);
    cell = cell * forget + ingate * change;
    hidden = outgate * std::tanh(cell);
    value += std::log(2.0 + std::exp(hidden)) - 0.1 * hidden;
  }
  return value / n;
}

double ba(int n, const double *x) {
  double wx = 0.03, wy = -0.04, wz = 0.02;
  const double theta = std::sqrt(wx * wx + wy * wy + wz * wz);
  const double ct = std::cos(theta), st = std::sin(theta);
  wx /= theta;
  wy /= theta;
  wz /= theta;
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = 3 * i;
    const double px = x[j] - 0.1;
    const double py = x[j + 1] + 0.2;
    const double pz = x[j + 2] + 3.0;
    const double dot = wx * px + wy * py + wz * pz;
    double qx = px * ct + (wy * pz - wz * py) * st + wx * dot * (1.0 - ct);
    double qy = py * ct + (wz * px - wx * pz) * st + wy * dot * (1.0 - ct);
    const double qz =
        pz * ct + (wx * py - wy * px) * st + wz * dot * (1.0 - ct);
    qx /= qz;
    qy /= qz;
    const double r2 = qx * qx + qy * qy;
    const double distortion = 1.0 + 0.01 * r2 - 0.001 * r2 * r2;
    qx = 800.0 * qx * distortion + 320.0;
    qy = 800.0 * qy * distortion + 240.0;
    value += (qx - 321.0) * (qx - 321.0) + (qy - 239.0) * (qy - 239.0);
  }
  return value / n;
}

double gmm(int n, const double *x) {
  constexpr int d = 4, components = 4;
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    double terms[components];
    for (int k = 0; k < components; ++k) {
      double total = 0.0;
      for (int j = 0; j < d; ++j) {
        const double centered = x[i * d + j] - 0.15 * (k - 1) * (j + 1);
        total += std::exp(0.02 * (j + 1) * (k + 1)) * centered * centered;
      }
      terms[k] = 0.1 * (k - 1) - 0.5 * total;
    }
    double maximum = terms[0];
    for (int k = 1; k < components; ++k)
      maximum = std::max(maximum, terms[k]);
    double sum = 0.0;
    for (double term : terms)
      sum += std::exp(term - maximum);
    value += std::log(sum) + maximum;
  }
  return -value / n;
}

double euler(int n, const double *x) {
  const double dt = 2.1 / n;
  double state = 1.0;
  for (int i = 0; i < n; ++i)
    state += dt * (-1.2 * state + 0.05 * x[i]);
  return state;
}

double rk4(int n, const double *x) {
  const double dt = 2.1 / n;
  double state = 1.0;
  for (int i = 0; i < n; ++i) {
    const double k1 = -1.2 * state + 0.05 * x[i];
    const double k2 = -1.2 * (state + 0.5 * dt * k1) + 0.05 * x[i];
    const double k3 = -1.2 * (state + 0.5 * dt * k2) + 0.05 * x[i];
    const double k4 = -1.2 * (state + dt * k3) + 0.05 * x[i];
    state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
  }
  return state;
}

double fft(int n, double *x) {
  for (int block = 2; block <= n; block *= 2) {
    const int half = block / 2;
    for (int i = 0; i < n; i += block) {
      for (int j = 0; j < half; ++j) {
        const double angle = -2.0 * std::acos(-1.0) * j / block;
        const double ar = x[i + j], ai = x[n + i + j];
        const double br = x[i + j + half], bi = x[n + i + j + half];
        const double tr = std::cos(angle) * br - std::sin(angle) * bi;
        const double ti = std::sin(angle) * br + std::cos(angle) * bi;
        x[i + j] = ar + tr;
        x[n + i + j] = ai + ti;
        x[i + j + half] = ar - tr;
        x[n + i + j + half] = ai - ti;
      }
    }
  }
  double value = 0.0;
  for (int i = 0; i < n; ++i)
    value += std::log(1.0 + x[i] * x[i] + x[n + i] * x[n + i]);
  value /= n;
  for (int block = n; block >= 2; block /= 2) {
    const int half = block / 2;
    for (int i = 0; i < n; i += block) {
      for (int j = 0; j < half; ++j) {
        const double angle = 2.0 * std::acos(-1.0) * j / block;
        const double ar = x[i + j], ai = x[n + i + j];
        const double br = x[i + j + half], bi = x[n + i + j + half];
        const double tr = 0.5 * (ar + br), ti = 0.5 * (ai + bi);
        const double dr = 0.5 * (ar - br), di = 0.5 * (ai - bi);
        x[i + j] = tr;
        x[n + i + j] = ti;
        x[i + j + half] = std::cos(angle) * dr - std::sin(angle) * di;
        x[n + i + j + half] = std::sin(angle) * dr + std::cos(angle) * di;
      }
    }
  }
  return value;
}

double brusselator(int n, const double *x) {
  const double alpha = 0.01 * (n - 1) * (n - 1);
  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    const int im = std::max(0, i - 1), ip = std::min(n - 1, i + 1);
    for (int j = 0; j < n; ++j) {
      const int jm = std::max(0, j - 1), jp = std::min(n - 1, j + 1);
      const int p = i * n + j;
      const double u = x[p], v = x[n * n + p], u2v = u * u * v;
      const double du = alpha * (x[im * n + j] + x[ip * n + j] + x[i * n + jm] +
                                 x[i * n + jp] - 4.0 * u) +
                        1.0 + u2v - 4.4 * u;
      const double dv =
          alpha * (x[n * n + im * n + j] + x[n * n + ip * n + j] +
                   x[n * n + i * n + jm] + x[n * n + i * n + jp] - 4.0 * v) +
          3.4 * u - u2v;
      value += du * du + dv * dv;
    }
  }
  return value / (n * n);
}

} // namespace

extern "C" double enzyme_study_cpp_objective(int workload, int n, double *x) {
  switch (workload) {
  case 1:
    return lstm(n, x);
  case 2:
    return ba(n, x);
  case 3:
    return gmm(n, x);
  case 4:
    return euler(n, x);
  case 5:
    return rk4(n, x);
  case 6:
    return fft(n, x);
  case 7:
    return brusselator(n, x);
  default:
    return 0.0;
  }
}

extern "C" double enzyme_study_cpp_primal(int workload, int n, double *x) {
  return enzyme_study_cpp_objective(workload, n, x);
}

extern "C" double enzyme_study_cpp_vjp(int workload, int n, double *x,
                                       double *x_bar) {
  return __enzyme_autodiff((void *)enzyme_study_cpp_objective, enzyme_const,
                           workload, enzyme_const, n, enzyme_dup, x, x_bar);
}
