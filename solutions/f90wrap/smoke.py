import numpy as np

import heat_model


model = heat_model.heat_model_m.new_heat_model(alpha=0.7, dx=0.2)
x = np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1
expected = x.copy()
expected[1:-1] += 0.35 * (x[:-2] - 2.0 * x[1:-1] + x[2:])
actual = np.empty_like(x)
model.step(x, 0.02, actual)
np.testing.assert_allclose(actual, expected, rtol=2.0e-14)
print("PASS")
