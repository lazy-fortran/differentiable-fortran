#!/usr/bin/env python3
"""Create the study figures from committed raw measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
plt.rcParams["svg.hashsalt"] = "differentiable-fortran-enzyme-readme"
COLORS = {
    "C++ Enzyme": "#3b5b92",
    "Flang Enzyme": "#d65f5f",
    "LFortran Enzyme": "#6a9f58",
    "JAX JIT": "#8a63a8",
}
MARKERS = {
    "C++ Enzyme": "o",
    "Flang Enzyme": "s",
    "LFortran Enzyme": "^",
    "JAX JIT": "D",
}
ORDER = list(COLORS)
WORKLOADS = ["LSTM", "BA", "GMM", "Euler", "RK4", "FFT", "Bruss"]


def save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(
        output / f"{name}.svg",
        bbox_inches="tight",
        metadata={"Date": "2026-07-15"},
    )
    svg = output / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    figure.savefig(output / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(figure)


def summaries(frame: pd.DataFrame) -> pd.DataFrame:
    steady = frame[frame["operation"].isin(("primal", "vjp"))]
    return (
        steady.groupby(["workload", "size", "input_length", "stack", "operation"])
        .time_us.agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def scaling(summary: pd.DataFrame, operation: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.5), sharey=False)
    for axis, workload in zip(axes.flat, WORKLOADS, strict=False):
        selected = summary[
            (summary["workload"] == workload) & (summary["operation"] == operation)
        ]
        for stack in ORDER:
            values = selected[selected["stack"] == stack].sort_values("input_length")
            axis.loglog(
                values.input_length,
                values["median"],
                marker=MARKERS[stack],
                color=COLORS[stack],
                label=stack,
                linewidth=1.8,
                markersize=4.5,
            )
            axis.fill_between(
                values.input_length,
                values.q25,
                values.q75,
                color=COLORS[stack],
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(workload)
        axis.grid(True, which="both", alpha=0.22)
        axis.set_xlabel("active input values")
        axis.set_ylabel("time per call (µs)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle(f"{operation.upper()} steady-state scaling", fontsize=15, y=1.01)
    figure.tight_layout()
    save(figure, output, f"{operation}_scaling")


def largest_size(summary: pd.DataFrame, output: Path) -> None:
    largest = summary.loc[
        summary.groupby("workload")["input_length"].transform("max")
        == summary["input_length"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    x = np.arange(len(WORKLOADS))
    width = 0.19
    for axis, operation in zip(axes, ("primal", "vjp"), strict=True):
        selected = largest[largest["operation"] == operation]
        for index, stack in enumerate(ORDER):
            values = (
                selected[selected["stack"] == stack]
                .set_index("workload")
                .reindex(WORKLOADS)
            )
            axis.bar(
                x + (index - 1.5) * width,
                values["median"],
                width,
                color=COLORS[stack],
                label=stack,
            )
        axis.set_yscale("log")
        axis.set_xticks(x, WORKLOADS, rotation=25, ha="right")
        axis.set_title(operation.upper())
        axis.grid(True, axis="y", which="both", alpha=0.22)
        axis.set_ylabel("time per call (µs, log scale)")
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Largest measured case for each workload", fontsize=15)
    figure.tight_layout()
    save(figure, output, "largest_case")


def overhead(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(
        index=["workload", "size", "input_length", "stack"],
        columns="operation",
        values="median",
    ).reset_index()
    pivot["ratio"] = pivot.vjp / pivot.primal
    largest = pivot.loc[
        pivot.groupby("workload")["input_length"].transform("max")
        == pivot["input_length"]
    ]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(WORKLOADS))
    width = 0.19
    for index, stack in enumerate(ORDER):
        values = (
            largest[largest["stack"] == stack].set_index("workload").reindex(WORKLOADS)
        )
        axis.bar(
            x + (index - 1.5) * width,
            values.ratio,
            width,
            color=COLORS[stack],
            label=stack,
        )
    axis.axhline(1.0, color="#333333", linewidth=1.0)
    axis.set_xticks(x, WORKLOADS)
    axis.set_ylabel("VJP time / primal time")
    axis.set_title("Reverse-mode overhead at the largest measured size")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(figure, output, "vjp_overhead")


def accuracy_plot(frame: pd.DataFrame, output: Path) -> None:
    maximum = (
        frame.groupby(["workload", "stack"])["vjp_max_abs_error"].max().reset_index()
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(WORKLOADS))
    width = 0.25
    for index, stack in enumerate(ORDER[:3]):
        values = (
            maximum[maximum["stack"] == stack].set_index("workload").reindex(WORKLOADS)
        )
        axis.bar(
            x + (index - 1) * width,
            np.maximum(values.vjp_max_abs_error, 1.0e-16),
            width,
            color=COLORS[stack],
            label=stack,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, WORKLOADS)
    axis.set_ylabel("maximum absolute gradient error")
    axis.set_title("Worst native Enzyme VJP error across the size sweep")
    axis.grid(True, axis="y", which="both", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, output, "vjp_accuracy")


def startup_plot(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["operation"].str.endswith("_startup")].copy()
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.5))
    for axis, workload in zip(axes.flat, WORKLOADS, strict=False):
        values = selected[selected["workload"] == workload]
        for operation, marker, color in (
            ("primal_startup", "o", "#3b5b92"),
            ("vjp_startup", "s", "#d65f5f"),
        ):
            line = values[values["operation"] == operation].sort_values("input_length")
            axis.loglog(
                line.input_length,
                line.time_us / 1.0e3,
                marker=marker,
                color=color,
                label=operation.replace("_", " "),
            )
        axis.set_title(workload)
        axis.set_xlabel("active input values")
        axis.set_ylabel("compile + first call (ms)")
        axis.grid(True, which="both", alpha=0.22)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle("JAX compilation and first execution", fontsize=15, y=1.01)
    figure.tight_layout()
    save(figure, output, "jax_startup")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=STUDY / "results/reference")
    parser.add_argument("--output", type=Path, default=STUDY / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings = pd.read_csv(args.results / "timings.csv")
    accuracy = pd.read_csv(args.results / "accuracy.csv")
    summary = summaries(timings)
    scaling(summary, "primal", args.output)
    scaling(summary, "vjp", args.output)
    largest_size(summary, args.output)
    overhead(summary, args.output)
    accuracy_plot(accuracy, args.output)
    startup_plot(timings, args.output)


if __name__ == "__main__":
    main()
