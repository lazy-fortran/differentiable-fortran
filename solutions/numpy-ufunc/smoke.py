import numpy as np

from heat_ufunc import heat_interior


x = np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1
actual = x.copy()
actual[1:-1] = heat_interior(x[:-2], x[1:-1], x[2:], 0.35)
expected = x.copy()
expected[1:-1] += 0.35 * (x[:-2] - 2.0 * x[1:-1] + x[2:])
np.testing.assert_allclose(actual, expected, rtol=2.0e-14)
assert isinstance(heat_interior, np.ufunc)
print("PASS")
