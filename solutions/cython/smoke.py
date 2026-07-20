import numpy as np

import cython_heat_step


x = np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1
expected = x.copy()
expected[1:-1] += 0.35 * (x[:-2] - 2.0 * x[1:-1] + x[2:])
np.testing.assert_allclose(
    cython_heat_step.primal(x, 0.7, 0.02, 0.2), expected, rtol=2.0e-14
)
print("PASS")
