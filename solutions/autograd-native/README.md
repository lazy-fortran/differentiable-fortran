# Python/NumPy Autograd

This baseline differentiates ordinary eager Python and NumPy-style operations
with [HIPS Autograd](https://github.com/HIPS/autograd). It performs no JIT
compilation and does not call the Fortran implementation.

```python
import numpy as np

from differentiable_fortran import autograd_native

x = np.linspace(0.0, 1.0, 64)
x_dot = np.ones_like(x)

y = autograd_native.heat_step(x, 0.7, 0.02, 0.2)
y, y_dot = autograd_native.jvp(
    x, 0.7, 0.02, 0.2, x_dot, 0.13, -0.01, 0.03
)
x_bar, alpha_bar, dt_bar, dx_bar = autograd_native.vjp(
    x, 0.7, 0.02, 0.2, np.ones_like(x)
)
```

The scaling plots label this implementation `Python/NumPy Autograd (eager)`.
Its timing includes Python tracing and tape construction on every derivative
evaluation, which is the behavior an eager user pays in a repeated workload.
