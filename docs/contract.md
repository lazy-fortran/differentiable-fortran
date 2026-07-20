# Common heat-step contract

All solutions implement one explicit finite-difference step of the one-dimensional
heat equation. For a state vector \(x\) of length \(n\), diffusivity \(\alpha\),
time step \(\Delta t\), and grid spacing \(\Delta x\), define

\[
    r = \frac{\alpha\,\Delta t}{\Delta x^2}.
\]

The boundary values are copied. For interior indices,

\[
    y_i = x_i + r (x_{i-1} - 2x_i + x_{i+1}).
\]

The public contract contains three operations: the primal map, its Jacobian-vector
product (JVP), and its vector-Jacobian product (VJP). Arrays contain IEEE binary64
values and use contiguous storage. `n` must be at least two and `dx` must be
nonzero. The examples use \(0 \le r \le 1/2\), the usual stability region for this
explicit scheme, although the routine does not enforce a physical time-step rule.

## Analytical JVP

For tangents \(\dot{x}\), \(\dot{\alpha}\), \(\dot{\Delta t}\), and
\(\dot{\Delta x}\),

\[
\dot r = \frac{\Delta t}{\Delta x^2}\dot\alpha
       + \frac{\alpha}{\Delta x^2}\dot{\Delta t}
       - \frac{2\alpha\Delta t}{\Delta x^3}\dot{\Delta x}.
\]

Writing \(L_i=x_{i-1}-2x_i+x_{i+1}\), the interior tangent is

\[
\dot y_i = \dot x_i + \dot r L_i
          + r(\dot x_{i-1}-2\dot x_i+\dot x_{i+1}).
\]

The boundary tangents are copied from \(\dot{x}\).

## Analytical VJP

Given an output cotangent \(w\), each interior component contributes
\(r w_i\), \((1-2r)w_i\), and \(r w_i\) to the three adjacent input
cotangents. Boundary outputs contribute directly to the corresponding boundary
inputs. If

\[
S = \sum_{i=2}^{n-1}w_i L_i,
\]

then the scalar cotangents are

\[
\bar\alpha = S\frac{\Delta t}{\Delta x^2},\qquad
\overline{\Delta t} = S\frac{\alpha}{\Delta x^2},\qquad
\overline{\Delta x} = -S\frac{2\alpha\Delta t}{\Delta x^3}.
\]

These expressions are the correctness oracle and an important performance
baseline. Automatic differentiation is not expected to beat a carefully written
closed-form derivative in every case.
