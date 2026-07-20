import numpy as np
import torch

from differentiable_fortran import heat_step_vjp
from heat_step_operator import heat_step


torch.set_default_dtype(torch.float64)
x = torch.tensor(np.sin(np.arange(12) * 0.3) + np.arange(12) * 0.1)
alpha = torch.tensor(0.7, requires_grad=True)
dt = torch.tensor(0.02, requires_grad=True)
dx = torch.tensor(0.2, requires_grad=True)
x.requires_grad_(True)
y_bar = torch.where(torch.arange(12) % 2 == 0, 1.0, -1.0) / (torch.arange(12) + 2.0)

compiled = torch.compile(heat_step, fullgraph=True)
actual = compiled(x, alpha, dt, dx)
(actual * y_bar).sum().backward()
expected = heat_step_vjp(
    x.detach().numpy(), alpha.item(), dt.item(), dx.item(), y_bar.numpy()
)
for actual_bar, expected_bar in zip(
    (x.grad, alpha.grad, dt.grad, dx.grad), expected, strict=True
):
    np.testing.assert_allclose(actual_bar.detach().numpy(), expected_bar, rtol=2.0e-14)
print("PASS")
