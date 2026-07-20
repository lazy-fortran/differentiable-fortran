# Benchmark method

Benchmarks answer three different questions and never combine their timings:

1. How expensive is one call across an interface?
2. How does the primal, JVP, or VJP scale with state-vector length?
3. What setup cost is paid for compilation, derivative generation, extension
   building, JIT compilation, or service startup?

Steady-state measurements warm each implementation before sampling. A sample
contains enough inner iterations to exceed timer resolution. No empty-loop
correction is applied. The plotted point is the median sample time and the interval
spans the 25th to 75th percentiles. Raw observations are retained in CSV, with host,
compiler, runtime, array size, operation, and warm-up policy recorded in a JSON
manifest.

JAX and PyTorch compilation is timed separately as compile plus first execution.
JAX's in-memory compilation cache is cleared first. PyTorch's compiler is reset and
each observation uses a fresh Inductor cache directory. PyTorch uses `fullgraph`,
the Inductor backend, and static shape specialization (`dynamic=False`) to match
JAX's per-shape compiled functions. The steady-state series executes an already
compiled function. PyTorch eager and Python/NumPy Autograd eager are separate
series and perform no JIT compilation.

For a workload making `k` calls, use `first_call + (k - 1) * steady_state` for a
compiled implementation and `k * eager` for an eager implementation. This keeps a
real compilation cost visible without charging it to every subsequent call.

Kernel and framework-startup plots are log-log: state-vector length is logarithmic
on the x-axis, and time per operation is logarithmic on the y-axis. Interface plots
use a categorical x-axis and nanoseconds per call on a logarithmic y-axis. Accuracy
plots show the relative 2-norm error against the analytical derivative; exact
agreement is reported separately because zero cannot appear on a logarithmic axis.

The reference framework timings use binary64 CPU arrays. Input construction is
outside the timed region, while output allocation, Python dispatch, tracing, tape
construction, and framework execution remain inside whenever the implementation
performs them on each call.

Run all available benchmarks with:

```console
uv run benchmark/run.py
uv run benchmark/plot.py
```

Unavailable optional implementations are reported as skipped and never replaced
with synthetic measurements. Reference figures committed to the repository are
identified by a machine-readable manifest and may be regenerated with
`benchmark/reproduce-reference.sh`.
