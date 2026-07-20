import numpy as np

import tesseract_api
from differentiable_fortran import heat_step, heat_step_jvp, heat_step_vjp


x = np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1
inputs = tesseract_api.InputSchema(x=x, alpha=0.7, dt=0.02, dx=0.2)
actual = tesseract_api.apply(inputs)
np.testing.assert_allclose(actual.y, heat_step(x, 0.7, 0.02, 0.2))

x_dot = np.cos(np.arange(12) * 0.2)
tangent = {"x": x_dot, "alpha": 0.13, "dt": -0.01, "dx": 0.03}
actual_jvp = tesseract_api.jacobian_vector_product(
    inputs, set(tangent), {"y"}, tangent
)["y"]
expected_jvp = heat_step_jvp(x, 0.7, 0.02, 0.2, x_dot, 0.13, -0.01, 0.03)[1]
np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2.0e-13)

y_bar = np.where(np.arange(12) % 2 == 0, 1.0, -1.0) / (np.arange(12) + 2.0)
actual_vjp = tesseract_api.vector_jacobian_product(
    inputs, {"x", "alpha", "dt", "dx"}, {"y"}, {"y": y_bar}
)
expected_vjp = heat_step_vjp(x, 0.7, 0.02, 0.2, y_bar)
for name, expected in zip(("x", "alpha", "dt", "dx"), expected_vjp, strict=True):
    np.testing.assert_allclose(actual_vjp[name], expected, rtol=2.0e-13)
print("PASS")
