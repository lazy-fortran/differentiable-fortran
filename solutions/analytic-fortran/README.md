# Analytical Fortran derivative

The analytical implementation is the correctness oracle and performance floor for
this problem. It implements the closed-form JVP and VJP from `docs/contract.md`
without a generated tape, runtime, or framework.

```console
fo test test_analytic
```

Use this solution when the formula remains maintainable and minimizing derivative
cost matters more than avoiding derivative source code.
