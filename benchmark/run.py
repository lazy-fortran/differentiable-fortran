#!/usr/bin/env python3
"""Collect reproducible kernel, interface, setup, and accuracy measurements."""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from importlib.metadata import version as package_version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from differentiable_fortran import NativeHeatStep, heat_step_jvp, heat_step_vjp
from differentiable_fortran import autograd_native, jax_native, torch_native


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark/results/reference"
SIZES = (16, 64, 256, 1024, 4096, 16384, 65536)
PARAMETERS = (0.7, 0.02, 0.2)
PARAMETER_DOT = (0.13, -0.01, 0.03)
REPEATS = 9
TARGET_SECONDS = 0.004
STARTUP_REPEATS = 1

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def version(command: list[str]) -> str:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return output.splitlines()[0]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def synchronize(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif torch.is_tensor(value) and value.device.type != "cpu":
        torch.cuda.synchronize(value.device)
    elif isinstance(value, (tuple, list)):
        for item in value:
            synchronize(item)
    return value


def measure(function: Callable[[], object]) -> tuple[list[float], int]:
    for _ in range(3):
        synchronize(function())
    loops = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(loops):
            synchronize(function())
        elapsed = (time.perf_counter_ns() - start) * 1.0e-9
        if elapsed >= TARGET_SECONDS or loops >= 1_048_576:
            break
        loops *= 2
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        for _ in range(loops):
            synchronize(function())
        samples.append((time.perf_counter_ns() - start) * 1.0e-9 / loops)
    return samples, loops


def first_call(function: Callable[[], object]) -> float:
    start = time.perf_counter_ns()
    synchronize(function())
    return (time.perf_counter_ns() - start) * 1.0e-9


def state(n: int):
    grid = np.arange(n, dtype=np.float64)
    x = np.sin(grid * 0.03) + grid * 1.0e-4
    x_dot = np.cos(grid * 0.02)
    y_bar = np.where(grid.astype(np.int64) % 2 == 0, 1.0, -1.0) / (grid + 2.0)
    return x, x_dot, y_bar


def append_samples(
    rows, benchmark, implementation, operation, n, samples, loops, note=""
):
    for sample, seconds in enumerate(samples):
        rows.append(
            {
                "benchmark": benchmark,
                "implementation": implementation,
                "operation": operation,
                "n": n,
                "sample": sample,
                "loops": loops,
                "time_seconds": f"{seconds:.17g}",
                "note": note,
            }
        )


def native_backends():
    return {
        "Fortran analytical": NativeHeatStep(
            ROOT / "build/libdifferentiable_fortran.so"
        ),
        "Flang + Enzyme": NativeHeatStep(
            ROOT / "build/flang-enzyme/libdf_flang_enzyme.so", enzyme=True
        ),
        "LFortran + Enzyme": NativeHeatStep(
            ROOT / "build/lfortran-enzyme/libdf_lfortran_enzyme.so", enzyme=True
        ),
    }


def collect_kernel(rows):
    native = native_backends()
    cache_root = ROOT / "build/benchmark-framework-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        x, x_dot, y_bar = state(n)
        for name, backend in native.items():
            calls = {
                "primal": lambda b=backend: b.primal(x, *PARAMETERS),
                "jvp": lambda b=backend: b.jvp(x, *PARAMETERS, x_dot, *PARAMETER_DOT)[
                    1
                ],
                "vjp": lambda b=backend: b.vjp(x, *PARAMETERS, y_bar)[0],
            }
            for operation, function in calls.items():
                samples, loops = measure(function)
                append_samples(rows, "kernel", name, operation, n, samples, loops)

        autograd_calls = {
            "primal": lambda: autograd_native.heat_step(x, *PARAMETERS),
            "jvp": lambda: autograd_native.jvp(x, *PARAMETERS, x_dot, *PARAMETER_DOT),
            "vjp": lambda: autograd_native.vjp(x, *PARAMETERS, y_bar),
        }
        for operation, function in autograd_calls.items():
            samples, loops = measure(function)
            append_samples(
                rows,
                "kernel",
                "Python/NumPy Autograd (eager)",
                operation,
                n,
                samples,
                loops,
            )

        jx = jnp.asarray(x)
        jx_dot = jnp.asarray(x_dot)
        jy_bar = jnp.asarray(y_bar)
        jp = tuple(jnp.asarray(value) for value in PARAMETERS)
        jpd = tuple(jnp.asarray(value) for value in PARAMETER_DOT)
        jax_functions = {
            "primal": jax_native.heat_step,
            "jvp": jax_native.jvp,
            "vjp": jax_native.vjp,
        }
        jax_arguments = {
            "primal": (jx, *jp),
            "jvp": (jx, *jp, jx_dot, *jpd),
            "vjp": (jx, *jp, jy_bar),
        }
        for operation, function in jax_functions.items():
            arguments = jax_arguments[operation]
            startup_samples = []
            for _ in range(STARTUP_REPEATS):
                jax.clear_caches()
                cold = jax.jit(function)
                startup_samples.append(first_call(lambda f=cold, a=arguments: f(*a)))
            append_samples(
                rows,
                "compile_first_call",
                "JAX native (jit)",
                operation,
                n,
                startup_samples,
                1,
                "in-memory cache cleared; compile plus first execution",
            )
            compiled = jax.jit(function)
            synchronize(compiled(*arguments))
            samples, loops = measure(lambda f=compiled, a=arguments: f(*a))
            append_samples(
                rows, "kernel", "JAX native (jit)", operation, n, samples, loops
            )

        tx = torch.from_numpy(x.copy())
        tx_dot = torch.from_numpy(x_dot.copy())
        ty_bar = torch.from_numpy(y_bar.copy())
        tp = tuple(torch.tensor(value) for value in PARAMETERS)
        tpd = tuple(torch.tensor(value) for value in PARAMETER_DOT)
        torch_functions = {
            "primal": torch_native.heat_step,
            "jvp": torch_native.jvp,
            "vjp": torch_native.vjp,
        }
        torch_arguments = {
            "primal": (tx, *tp),
            "jvp": (tx, *tp, tx_dot, *tpd),
            "vjp": (tx, *tp, ty_bar),
        }
        for operation, function in torch_functions.items():
            arguments = torch_arguments[operation]
            samples, loops = measure(lambda f=function, a=arguments: f(*a))
            append_samples(
                rows,
                "kernel",
                "PyTorch native (eager)",
                operation,
                n,
                samples,
                loops,
            )

            startup_samples = []
            for _ in range(STARTUP_REPEATS):
                torch.compiler.reset()
                with tempfile.TemporaryDirectory(
                    prefix="torchinductor-", dir=cache_root
                ) as cache:
                    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                    try:
                        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
                        cold = torch.compile(
                            function,
                            fullgraph=True,
                            dynamic=False,
                            backend="inductor",
                        )
                        startup_samples.append(
                            first_call(lambda f=cold, a=arguments: f(*a))
                        )
                    finally:
                        if previous_cache is None:
                            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                        else:
                            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache
            append_samples(
                rows,
                "compile_first_call",
                "PyTorch native (compile)",
                operation,
                n,
                startup_samples,
                1,
                "compiler reset; fresh Inductor cache; compile plus first execution",
            )

            torch.compiler.reset()
            compiled = torch.compile(
                function, fullgraph=True, dynamic=False, backend="inductor"
            )
            synchronize(compiled(*arguments))
            samples, loops = measure(lambda f=compiled, a=arguments: f(*a))
            append_samples(
                rows, "kernel", "PyTorch native (compile)", operation, n, samples, loops
            )


def import_solution(directory: str, module: str):
    sys.path.insert(0, str(ROOT / "solutions" / directory))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


def import_path(directory: Path, module: str):
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.pop(0)


def collect_interface(rows):
    n = 8
    x, _, _ = state(n)
    native = native_backends()["Fortran analytical"]
    calls = {"ctypes": lambda: native.primal(x, *PARAMETERS)}

    cython = import_solution("cython", "cython_heat_step")
    calls["Cython"] = lambda: cython.primal(x, *PARAMETERS)

    f90wrap = import_path(ROOT / "build/f90wrap", "heat_model")
    model = f90wrap.heat_model_m.new_heat_model(alpha=PARAMETERS[0], dx=PARAMETERS[2])
    f90wrap_output = np.empty_like(x)
    calls["f90wrap"] = lambda: model.step(x, PARAMETERS[1], f90wrap_output)

    ufunc = import_solution("numpy-ufunc", "heat_ufunc")
    calls["NumPy ufunc"] = lambda: ufunc.heat_interior(
        x[:-2], x[1:-1], x[2:], PARAMETERS[0] * PARAMETERS[1] / PARAMETERS[2] ** 2
    )

    jax_ffi = import_solution("jax-ffi", "heat_step_ffi")
    jx = jnp.asarray(x)
    jax_call = jax.jit(jax_ffi.primal)
    synchronize(jax_call(jx, *PARAMETERS))
    calls["JAX FFI (jit)"] = lambda: jax_call(jx, *PARAMETERS)

    torch_operator = import_solution("pytorch-custom-op", "heat_step_operator")
    tx = torch.from_numpy(x.copy())
    tp = tuple(torch.tensor(value) for value in PARAMETERS)
    torch_call = torch.compile(
        torch_operator.heat_step, fullgraph=True, dynamic=False, backend="inductor"
    )
    torch_call(tx, *tp)
    calls["PyTorch custom op"] = lambda: torch_call(tx, *tp)

    os.environ["DF_TESSERACT_LIBRARY"] = str(
        ROOT / "build/lfortran-enzyme/libdf_lfortran_enzyme.so"
    )
    tesseract = import_solution("tesseract", "tesseract_api")
    tesseract_inputs = tesseract.InputSchema(
        x=x, alpha=PARAMETERS[0], dt=PARAMETERS[1], dx=PARAMETERS[2]
    )
    calls["Tesseract in-process"] = lambda: tesseract.apply(tesseract_inputs)

    notes = {
        "ctypes": "allocates output",
        "Cython": "allocates output",
        "f90wrap": "reuses output",
        "NumPy ufunc": "interior update; allocates output",
        "JAX FFI (jit)": "device input reused",
        "PyTorch custom op": "tensor input reused",
        "Tesseract in-process": "validated input reused; validates output",
    }
    for name, function in calls.items():
        samples, loops = measure(function)
        append_samples(
            rows, "interface", name, "primal", n, samples, loops, notes[name]
        )


def relative_error(actual, expected) -> float:
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(np.asarray(actual) - expected) / denominator)


def collect_accuracy(path: Path):
    n = 4096
    x, x_dot, y_bar = state(n)
    expected_jvp = heat_step_jvp(x, *PARAMETERS, x_dot, *PARAMETER_DOT)[1]
    expected_vjp = heat_step_vjp(x, *PARAMETERS, y_bar)
    records = []
    for name, backend in native_backends().items():
        records.append(
            (
                name,
                "jvp",
                relative_error(
                    backend.jvp(x, *PARAMETERS, x_dot, *PARAMETER_DOT)[1], expected_jvp
                ),
            )
        )
        actual_vjp = backend.vjp(x, *PARAMETERS, y_bar)
        actual_flat = np.concatenate((actual_vjp[0], np.asarray(actual_vjp[1:])))
        expected_flat = np.concatenate((expected_vjp[0], np.asarray(expected_vjp[1:])))
        records.append((name, "vjp", relative_error(actual_flat, expected_flat)))

    actual_autograd_jvp = autograd_native.jvp(x, *PARAMETERS, x_dot, *PARAMETER_DOT)[1]
    actual_autograd_vjp = autograd_native.vjp(x, *PARAMETERS, y_bar)
    records.append(
        (
            "Python/NumPy Autograd (eager)",
            "jvp",
            relative_error(actual_autograd_jvp, expected_jvp),
        )
    )
    records.append(
        (
            "Python/NumPy Autograd (eager)",
            "vjp",
            relative_error(
                np.concatenate(
                    (
                        np.asarray(actual_autograd_vjp[0]),
                        np.asarray(actual_autograd_vjp[1:]).reshape(-1),
                    )
                ),
                np.concatenate((expected_vjp[0], np.asarray(expected_vjp[1:]))),
            ),
        )
    )

    jx = jnp.asarray(x)
    jp = tuple(jnp.asarray(value) for value in PARAMETERS)
    actual_jax_jvp = jax_native.jvp(
        jx,
        *jp,
        jnp.asarray(x_dot),
        *(jnp.asarray(value) for value in PARAMETER_DOT),
    )[1]
    actual_jax_vjp = jax_native.vjp(jx, *jp, jnp.asarray(y_bar))
    records.append(
        ("JAX native (jit)", "jvp", relative_error(actual_jax_jvp, expected_jvp))
    )
    records.append(
        (
            "JAX native (jit)",
            "vjp",
            relative_error(
                np.concatenate(
                    (np.asarray(actual_jax_vjp[0]), np.asarray(actual_jax_vjp[1:]))
                ),
                np.concatenate((expected_vjp[0], np.asarray(expected_vjp[1:]))),
            ),
        )
    )

    tx = torch.from_numpy(x.copy())
    tp = tuple(torch.tensor(value) for value in PARAMETERS)
    actual_torch_jvp = torch_native.jvp(
        tx,
        *tp,
        torch.from_numpy(x_dot.copy()),
        *(torch.tensor(value) for value in PARAMETER_DOT),
    )[1]
    actual_torch_vjp = torch_native.vjp(tx, *tp, torch.from_numpy(y_bar.copy()))
    records.append(
        (
            "PyTorch native (compile)",
            "jvp",
            relative_error(actual_torch_jvp, expected_jvp),
        )
    )
    records.append(
        (
            "PyTorch native (compile)",
            "vjp",
            relative_error(
                np.concatenate(
                    (
                        actual_torch_vjp[0].numpy(),
                        np.asarray([value.item() for value in actual_torch_vjp[1:]]),
                    )
                ),
                np.concatenate((expected_vjp[0], np.asarray(expected_vjp[1:]))),
            ),
        )
    )

    epsilon = 1.0e-7
    plus = native_backends()["Fortran analytical"].primal(
        x + epsilon * x_dot,
        *(np.asarray(PARAMETERS) + epsilon * np.asarray(PARAMETER_DOT)),
    )
    minus = native_backends()["Fortran analytical"].primal(
        x - epsilon * x_dot,
        *(np.asarray(PARAMETERS) - epsilon * np.asarray(PARAMETER_DOT)),
    )
    records.append(
        (
            "Centered finite difference",
            "jvp",
            relative_error((plus - minus) / (2 * epsilon), expected_jvp),
        )
    )

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["implementation", "operation", "relative_l2_error"],
            lineterminator="\n",
        )
        writer.writeheader()
        for implementation, operation, error in records:
            writer.writerow(
                {
                    "implementation": implementation,
                    "operation": operation,
                    "relative_l2_error": f"{error:.17g}",
                }
            )


def collect_toolchains(path: Path):
    records = []
    for name, script, library in (
        ("Flang + Enzyme", "toolchains/flang-enzyme/build.sh", "libdf_flang_enzyme.so"),
        (
            "LFortran + Enzyme",
            "toolchains/lfortran-enzyme/build.sh",
            "libdf_lfortran_enzyme.so",
        ),
    ):
        for sample in range(5):
            output = (
                ROOT
                / "build/benchmark-toolchains"
                / name.lower().replace(" ", "-").replace("+", "")
                / str(sample)
            )
            start = time.perf_counter_ns()
            subprocess.run(
                [str(ROOT / script)],
                cwd=ROOT,
                env={**os.environ, "BUILD_DIR": str(output)},
                check=True,
                stdout=subprocess.DEVNULL,
            )
            seconds = (time.perf_counter_ns() - start) * 1.0e-9
            records.append(
                {
                    "implementation": name,
                    "sample": sample,
                    "build_seconds": f"{seconds:.17g}",
                    "library_bytes": (output / library).stat().st_size,
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=records[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    collect_kernel(rows)
    collect_interface(rows)
    with (RESULTS / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    collect_accuracy(RESULTS / "accuracy.csv")
    collect_toolchains(RESULTS / "toolchains.csv")
    manifest = {
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.system(),
        "processor": cpu_model(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "autograd": package_version("autograd"),
        "jax": jax.__version__,
        "torch": torch.__version__,
        "flang": version(["flang-new", "--version"]),
        "lfortran": version([os.environ.get("LFORTRAN", "lfortran"), "--version"]),
        "opt": version(["opt", "--version"]),
        "enzyme_plugin": Path(os.environ.get("ENZYME_PLUGIN", "LLVMEnzyme.so")).name,
        "sizes": SIZES,
        "repeats": REPEATS,
        "startup_repeats": STARTUP_REPEATS,
        "target_sample_seconds": TARGET_SECONDS,
        "jax_device": str(jax.devices()[0]),
        "jax_compilation_cache_dir": jax.config.jax_compilation_cache_dir,
        "torch_device": "cpu",
        "torch_compile": {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": True,
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
