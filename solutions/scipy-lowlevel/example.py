from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
from scipy import LowLevelCallable, integrate


ROOT = Path(__file__).resolve().parents[2]
library = ctypes.CDLL(str(ROOT / "build/libdifferentiable_fortran.so"))
gaussian = library.df_gaussian
gaussian.argtypes = [ctypes.c_double]
gaussian.restype = ctypes.c_double

# ctypes establishes the signature and exposes the address. QUADPACK calls that
# address directly during integration; Python and ctypes are not in the hot loop.
callback = LowLevelCallable(gaussian)
value, error = integrate.quad(callback, -1.0, 1.0)
expected = math.sqrt(math.pi) * math.erf(1.0)
np.testing.assert_allclose(value, expected, rtol=2.0e-14)
print(f"integral={value:.16g}, estimated error={error:.3g}")
