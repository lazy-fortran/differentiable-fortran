# Native JAX

The JAX baseline expresses the heat step with JAX array primitives and obtains
JVPs and VJPs from `jax.jvp` and `jax.vjp`. Under `jax.jit`, XLA can inspect and
fuse the complete computation.

```python
from jax import jit
from differentiable_fortran.jax_native import heat_step

compiled_step = jit(heat_step)
y = compiled_step(x, alpha, dt, dx)
```

Benchmarks report first-call compilation and steady-state execution separately.
