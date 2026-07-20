# f90wrap for a modern object-oriented Fortran API

This solution wraps `heat_model_t`, a modern Fortran derived type with private
state and type-bound procedures. Python users construct a model, inspect or change
its parameters, and invoke its methods directly:

```python
model = heat_model.heat_model_m.new_heat_model(alpha=0.7, dx=0.2)
model.step(x, dt, y)
model.set_alpha(0.5)
```

Build and test it with:

```console
uv sync --extra wrappers
solutions/f90wrap/build.sh
```

f90wrap maps the derived type and its methods to a Python class. Framework FFI
layers require a smaller C-compatible kernel, so a library can ship both APIs.
