import jax
import jax.numpy as jnp
import numpy as np

import heat_step_ffi
from differentiable_fortran import heat_step_jvp, heat_step_vjp


x = jnp.asarray(np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1)
x_dot = jnp.cos(jnp.arange(12) * 0.2)
y_bar = jnp.where(jnp.arange(12) % 2 == 0, 1.0, -1.0) / (jnp.arange(12) + 2.0)
parameters = (0.7, 0.02, 0.2)
parameter_dot = (0.13, -0.01, 0.03)

compiled_jvp = jax.jit(heat_step_ffi.jvp)
actual_y, actual_y_dot = compiled_jvp(x, *parameters, x_dot, *parameter_dot)
expected_y, expected_y_dot = heat_step_jvp(
    np.asarray(x), *parameters, np.asarray(x_dot), *parameter_dot
)
np.testing.assert_allclose(actual_y, expected_y, rtol=2.0e-14)
np.testing.assert_allclose(actual_y_dot, expected_y_dot, rtol=2.0e-14)

compiled_vjp = jax.jit(heat_step_ffi.vjp)
actual_vjp = compiled_vjp(x, *parameters, y_bar)
expected_vjp = heat_step_vjp(np.asarray(x), *parameters, np.asarray(y_bar))
for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14)
print("PASS")
