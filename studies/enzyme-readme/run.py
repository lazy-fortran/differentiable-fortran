#!/usr/bin/env python3
"""Validate and benchmark all seven workloads across the four pipelines."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from jax_kernels import WORKLOADS, compiled

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BUILD = ROOT / "build/enzyme-readme"
SIZES = {
    1: (16, 64, 256, 1024),
    2: (8, 32, 128, 512),
    3: (8, 32, 128, 512),
    4: (16, 64, 256, 1024),
    5: (16, 64, 256, 1024),
    6: (16, 64, 256, 1024),
    7: (8, 16, 32, 64),
}


def processor_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def input_length(workload: int, n: int) -> int:
    return {1: n, 2: 3 * n, 3: 4 * n, 4: n, 5: n, 6: 2 * n, 7: 2 * n * n}[workload]


def make_input(workload: int, n: int) -> np.ndarray:
    length = input_length(workload, n)
    indices = np.arange(length, dtype=np.float64)
    if workload == 7:
        grid = np.arange(n * n, dtype=np.float64)
        u = 1.2 + 0.1 * np.sin(0.13 * grid)
        v = 2.8 + 0.1 * np.cos(0.17 * grid)
        return np.ascontiguousarray(np.concatenate((u, v)))
    return np.ascontiguousarray(0.2 * np.sin(0.13 * indices) + 0.01 * np.cos(indices))


class Native:
    def __init__(self, name: str, path: Path, prefix: str):
        self.name = name
        library = ctypes.CDLL(str(path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.primal = getattr(library, f"{prefix}_primal")
        self.primal.argtypes = [ctypes.c_int, ctypes.c_int, pointer]
        self.primal.restype = ctypes.c_double
        self.vjp = getattr(library, f"{prefix}_vjp")
        self.vjp.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer]
        self.vjp.restype = ctypes.c_double


def timed(function, target: float = 0.03, samples: int = 9) -> tuple[int, list[float]]:
    loops = 1
    while True:
        started = time.perf_counter()
        for _ in range(loops):
            function()
        elapsed = time.perf_counter() - started
        if elapsed >= target or loops >= 1 << 18:
            break
        loops *= 2
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(loops):
            function()
        values.append((time.perf_counter_ns() - started) / loops / 1.0e3)
    return loops, values


def native_stacks() -> list[Native]:
    return [
        Native("C++ Enzyme", BUILD / "libstudy_cpp_enzyme.so", "enzyme_study_cpp"),
        Native("Flang Enzyme", BUILD / "libstudy_flang_enzyme.so", "enzyme_study"),
        Native(
            "LFortran Enzyme", BUILD / "libstudy_lfortran_enzyme.so", "enzyme_study"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY / "results/reference")
    parser.add_argument(
        "--quick", action="store_true", help="Use only the two middle sizes"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stacks = native_stacks()
    rows, accuracy = [], []
    for workload, (name, _) in WORKLOADS.items():
        sizes = SIZES[workload][1:3] if args.quick else SIZES[workload]
        for n in sizes:
            original = make_input(workload, n)
            jax_primal, jax_vjp = compiled(workload, n)
            device_input = jax.device_put(original)

            compile_started = time.perf_counter()
            jax_value = float(jax_primal(device_input).block_until_ready())
            primal_startup = time.perf_counter() - compile_started
            compile_started = time.perf_counter()
            jax_gradient = np.asarray(jax_vjp(device_input).block_until_ready())
            vjp_startup = time.perf_counter() - compile_started

            reference_value, reference_gradient = jax_value, jax_gradient
            for stack in stacks:
                x = original.copy()
                value = stack.primal(workload, n, x)
                gradient = np.zeros_like(x)
                stack.vjp(workload, n, x, gradient)
                accuracy.append(
                    {
                        "workload": name,
                        "size": n,
                        "stack": stack.name,
                        "primal_abs_error": abs(value - reference_value),
                        "vjp_max_abs_error": float(
                            np.max(np.abs(gradient - reference_gradient))
                        ),
                        "vjp_max_rel_error": float(
                            np.max(
                                np.abs(gradient - reference_gradient)
                                / np.maximum(1.0e-14, np.abs(reference_gradient))
                            )
                        ),
                    }
                )
                if not np.isclose(value, reference_value, rtol=2e-10, atol=2e-10):
                    raise RuntimeError(f"{stack.name} {name} n={n}: primal mismatch")
                if not np.allclose(gradient, reference_gradient, rtol=3e-8, atol=3e-9):
                    raise RuntimeError(f"{stack.name} {name} n={n}: VJP mismatch")

                def native_primal(stack=stack, x=x):
                    stack.primal(workload, n, x)

                def native_vjp(stack=stack, x=x, gradient=gradient):
                    gradient.fill(0.0)
                    stack.vjp(workload, n, x, gradient)

                for operation, function in (
                    ("primal", native_primal),
                    ("vjp", native_vjp),
                ):
                    loops, values = timed(function)
                    for sample, value_us in enumerate(values):
                        rows.append(
                            {
                                "workload": name,
                                "size": n,
                                "input_length": len(x),
                                "stack": stack.name,
                                "operation": operation,
                                "sample": sample,
                                "loops": loops,
                                "time_us": value_us,
                            }
                        )

            def run_jax_primal():
                jax_primal(device_input).block_until_ready()

            def run_jax_vjp():
                jax_vjp(device_input).block_until_ready()

            for operation, function, startup in (
                ("primal", run_jax_primal, primal_startup),
                ("vjp", run_jax_vjp, vjp_startup),
            ):
                loops, values = timed(function)
                for sample, value_us in enumerate(values):
                    rows.append(
                        {
                            "workload": name,
                            "size": n,
                            "input_length": len(original),
                            "stack": "JAX JIT",
                            "operation": operation,
                            "sample": sample,
                            "loops": loops,
                            "time_us": value_us,
                        }
                    )
                rows.append(
                    {
                        "workload": name,
                        "size": n,
                        "input_length": len(original),
                        "stack": "JAX JIT",
                        "operation": f"{operation}_startup",
                        "sample": 0,
                        "loops": 1,
                        "time_us": startup * 1.0e6,
                    }
                )
            print(f"validated and timed {name:5s} n={n:4d}", flush=True)

    for filename, data in (("timings.csv", rows), ("accuracy.csv", accuracy)):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": processor_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "numpy": np.__version__,
        "enzyme_benchmarks_commit": "9fa1579296b28d9642f8b69a6fffb753211c3f88",
        "versions": (BUILD / "versions.txt").read_text().splitlines(),
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
        ).stdout.strip(),
        "samples": 9,
        "statistic": "median with q25-q75 plotted",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
