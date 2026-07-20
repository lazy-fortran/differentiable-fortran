# ISO C plus ctypes

This is the shortest Python path to the minimal C-compatible Fortran API. It is a
good default when each call performs enough numerical work that a Python foreign
function call is negligible.

```python
from differentiable_fortran import NativeHeatStep

model = NativeHeatStep("build/libdifferentiable_fortran.so")
y = model.primal(x, alpha=0.7, dt=0.02, dx=0.2)
```

`NativeHeatStep` declares all argument types and accepts only contiguous
`float64` arrays. It does not copy arrays that already satisfy the contract.

`ctypes` is not the best path for a tiny function called repeatedly by Python.
Batch work into a coarser Fortran routine or use one of the compiled-loop
integrations. In the SciPy example, `ctypes` obtains a function address but is not
in the integration loop.
