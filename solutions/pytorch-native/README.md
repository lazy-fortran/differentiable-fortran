# Native PyTorch

The PyTorch baseline expresses the heat step with tensor operations and obtains
JVPs and VJPs from `torch.func`. `torch.compile` is measured with first-call
compilation separated from steady-state execution.

This baseline measures the cost of the tensor implementation under the same
contract as the compiled Fortran kernels.
