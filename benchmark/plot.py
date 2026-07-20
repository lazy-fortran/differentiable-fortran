#!/usr/bin/env python3
"""Render reference benchmark figures from committed raw measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark/results/reference"
FIGURES = ROOT / "benchmark/figures"
COLORS = {
    "Fortran analytical": "#0072B2",
    "Flang + Enzyme": "#D55E00",
    "LFortran + Enzyme": "#E69F00",
    "JAX native (jit)": "#009E73",
    "PyTorch native (compile)": "#CC79A7",
    "PyTorch native (eager)": "#56B4E9",
    "Python/NumPy Autograd (eager)": "#333333",
}
MARKERS = {
    name: marker
    for name, marker in zip(COLORS, ("o", "s", "^", "D", "v", "P", "X"), strict=True)
}
plt.rcParams["svg.hashsalt"] = "differentiable-fortran"


def size_label(value, _position):
    if value >= 1024:
        return f"{value / 1024:g}k"
    return f"{value:g}"


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{name}.svg"
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def summary(data, group):
    grouped = data.groupby(group, as_index=False)["time_seconds"]
    result = grouped.median().rename(columns={"time_seconds": "median"})
    result["q25"] = grouped.quantile(0.25)["time_seconds"]
    result["q75"] = grouped.quantile(0.75)["time_seconds"]
    return result


def scaling(timings, operation):
    data = timings[(timings.benchmark == "kernel") & (timings.operation == operation)]
    data = summary(data, ["implementation", "n"])
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for name in COLORS:
        group = data[data.implementation == name]
        if group.empty:
            continue
        axis.plot(
            group.n,
            group["median"],
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=1.8,
            markersize=4,
        )
        axis.fill_between(
            group.n, group.q25, group.q75, color=COLORS[name], alpha=0.16, linewidth=0
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("State-vector length, n")
    axis.set_ylabel(f"{operation.upper()} time [s]")
    axis.set_xticks(sorted(data.n.unique()))
    axis.xaxis.set_major_formatter(FuncFormatter(size_label))
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    figure.tight_layout()
    save(figure, f"{operation}_scaling")


def derivative_ratio(timings):
    data = summary(
        timings[timings.benchmark == "kernel"], ["implementation", "operation", "n"]
    )
    pivot = data.pivot(
        index=["implementation", "n"], columns="operation", values="median"
    ).reset_index()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for name in COLORS:
        group = pivot[pivot.implementation == name]
        if group.empty:
            continue
        axis.plot(
            group.n,
            group.vjp / group.primal,
            label=name,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=1.8,
            markersize=4,
        )
    axis.axhline(1.0, color="0.35", linewidth=1, linestyle="--")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("State-vector length, n")
    axis.set_ylabel("VJP time / primal time")
    axis.set_xticks(sorted(pivot.n.unique()))
    axis.xaxis.set_major_formatter(FuncFormatter(size_label))
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    figure.tight_layout()
    save(figure, "vjp_primal_ratio")


def framework_startup(timings):
    data = timings[timings.benchmark == "compile_first_call"]
    data = summary(data, ["implementation", "operation", "n"])
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.4), sharex=True, sharey=True)
    for axis, operation in zip(axes, ("primal", "jvp", "vjp"), strict=True):
        selected = data[data.operation == operation]
        for name in ("JAX native (jit)", "PyTorch native (compile)"):
            group = selected[selected.implementation == name]
            axis.plot(
                group.n,
                group["median"],
                label=name,
                color=COLORS[name],
                marker=MARKERS[name],
                linewidth=1.8,
                markersize=4,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(operation.upper())
        axis.set_xlabel("State length, n")
        axis.set_xticks(sorted(selected.n.unique())[::2])
        axis.xaxis.set_major_formatter(FuncFormatter(size_label))
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Compile + first execution [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    figure.tight_layout()
    save(figure, "framework_startup")


def interface(timings):
    data = summary(timings[timings.benchmark == "interface"], ["implementation"])
    data = data.sort_values("median")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(data))
    axis.bar(x, data["median"] * 1.0e9, color="#56B4E9")
    axis.errorbar(
        x,
        data["median"] * 1.0e9,
        yerr=[(data["median"] - data.q25) * 1.0e9, (data.q75 - data["median"]) * 1.0e9],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axis.set_yscale("log")
    axis.set_ylabel("N=8 primal call [ns]")
    axis.set_xticks(x, data.implementation, rotation=28, ha="right")
    axis.grid(True, axis="y", which="both", alpha=0.22)
    figure.tight_layout()
    save(figure, "interface_latency")


def accuracy():
    data = pd.read_csv(RESULTS / "accuracy.csv")
    data = data[data.implementation != "Fortran analytical"].copy()
    floor = 1.0e-18
    data["plotted"] = data.relative_l2_error.clip(lower=floor)
    names = list(dict.fromkeys(data.implementation))
    positions = {name: index for index, name in enumerate(reversed(names))}
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for operation, marker, color in (
        ("jvp", "o", "#0072B2"),
        ("vjp", "s", "#D55E00"),
    ):
        selected = data[data.operation == operation]
        axis.scatter(
            selected.plotted,
            [positions[name] for name in selected.implementation],
            label=operation.upper(),
            marker=marker,
            color=color,
            s=42,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Relative L2 error against analytical derivative")
    axis.set_yticks(
        range(len(names)),
        [name.replace("Python/NumPy ", "") for name in reversed(names)],
    )
    axis.grid(True, axis="x", which="both", alpha=0.22)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    save(figure, "derivative_accuracy")


def toolchains():
    data = pd.read_csv(RESULTS / "toolchains.csv")
    timing = data.groupby("implementation").build_seconds.agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    timing.columns = ["median", "q25", "q75"]
    size = data.groupby("implementation").library_bytes.median() / 1024
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    names = list(timing.index)
    positions = np.arange(len(names))
    axes[0].bar(positions, timing["median"], color=[COLORS[name] for name in names])
    axes[0].errorbar(
        positions,
        timing["median"],
        yerr=[timing["median"] - timing.q25, timing.q75 - timing["median"]],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].set(
        ylabel="Derivative build pipeline [s]", xticks=positions, xticklabels=names
    )
    axes[1].bar(positions, size[names], color=[COLORS[name] for name in names])
    axes[1].set(ylabel="Shared library [KiB]", xticks=positions, xticklabels=names)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.22)
    figure.tight_layout()
    save(figure, "compiler_toolchains")


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    timings = pd.read_csv(RESULTS / "timings.csv")
    for operation in ("primal", "jvp", "vjp"):
        scaling(timings, operation)
    derivative_ratio(timings)
    framework_startup(timings)
    interface(timings)
    accuracy()
    toolchains()
    print("Rendered reference figures")


if __name__ == "__main__":
    main()
