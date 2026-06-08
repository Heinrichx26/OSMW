from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r41_observable_cost import (
    Label,
    build_terminal_columns,
    exact_sparse_task_p2,
    run_terminal_collision_estimator,
    sample_all_labels,
    terminal_observable_truncation_errors,
    terminal_truncation_profile,
)
from r41_path_entropy import Gate, gate, pauli_transfer_matrix, unitary_from_gates
from r41_task_resolved_checks import task_entropy_from_transfer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41131


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        workload_rows = read_csv(DATA_DIR / "r41_stress_workloads.csv")
        estimator_rows = read_csv(DATA_DIR / "r41_stress_estimators.csv")
        plot_stress(workload_rows, estimator_rows)
        return

    validation = run_quick_validation()
    (DATA_DIR / "r41_stress_workloads_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.quick:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    workload_rows = run_local_window_stress()
    estimator_rows, truncation_rows = run_estimator_stress()
    write_csv(DATA_DIR / "r41_stress_workloads.csv", workload_rows)
    write_csv(DATA_DIR / "r41_stress_estimators.csv", estimator_rows)
    write_csv(DATA_DIR / "r41_stress_truncation.csv", truncation_rows)
    summary = summarize(workload_rows, estimator_rows, truncation_rows)
    (DATA_DIR / "r41_stress_workloads.json").write_text(
        json.dumps(
            {
                "validation": validation,
                "summary": summary,
                "workload_rows": workload_rows,
                "estimator_rows": estimator_rows,
                "truncation_rows": truncation_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_stress(workload_rows, estimator_rows)


def run_quick_validation() -> dict[str, object]:
    n_qubits = 4
    width = 2
    gates = stress_channel(n_qubits, depth=2)
    labels = window_xy_full_labels(n_qubits, width)
    transfer = pauli_transfer_matrix(unitary_from_gates(n_qubits, gates))
    exact = task_entropy_from_transfer(transfer, [label_to_index(label) for label in labels])
    sparse = exact_sparse_task_p2(n_qubits, gates, labels, max_terms=30_000)
    terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=30_000)
    profile = terminal_truncation_profile(terminal)
    all_k90 = math.ceil(0.9 * (4**n_qubits))
    local_window_ratio = float(profile["k90"]) / all_k90
    checks = [
        {
            "case": "exact_ptm_vs_sparse_window_xy",
            "observed": sparse,
            "expected": exact,
            "abs_error": abs(sparse - exact),
            "passed": abs(sparse - exact) < 1e-10,
        },
        {
            "case": "local_window_retained_support_below_all_pauli",
            "observed_ratio": local_window_ratio,
            "passed": local_window_ratio < 1.0,
        },
        {
            "case": "no_sparse_cap",
            "capped_labels": int(terminal["capped_labels"]),
            "passed": int(terminal["capped_labels"]) == 0,
        },
    ]
    return {
        "case": "local_window_scaling_quick_validation",
        "checks": checks,
        "max_abs_error": max(float(item.get("abs_error", 0.0)) for item in checks),
        "local_window_ratio": local_window_ratio,
        "passed": all(bool(item["passed"]) for item in checks),
    }


def run_local_window_stress(
    n_values: list[int] | None = None,
    widths: list[int] | None = None,
    max_terms: int = 250_000,
) -> list[dict[str, object]]:
    if n_values is None:
        n_values = [16, 32, 64, 96]
    if widths is None:
        widths = [1, 2, 3, 4, 5]
    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        gates = stress_channel(n_qubits, depth=2)
        all_row = all_pauli_row(n_qubits)
        for width in widths:
            specs = [
                ("window_z_string", window_z_string_labels(n_qubits, width)),
                ("window_x_string", window_x_string_labels(n_qubits, width)),
                ("window_xy_ensemble", window_xy_full_labels(n_qubits, width)),
            ]
            for task, labels in specs:
                terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=max_terms)
                profile = terminal_truncation_profile(terminal)
                rows.append(
                    {
                        "family": "local_window_stress",
                        "n_qubits": n_qubits,
                        "window_width": width,
                        "task": task,
                        "task_description": task_description(task),
                        "operator_magic": math.nan,
                        "task_p2": profile["task_p2"],
                        "exp_task_p2": profile["exp_task_p2"],
                        "k90": float(profile["k90"]),
                        "log10_k90": math.log10(max(float(profile["k90"]), 1e-300)),
                        "k90_per_window": float(profile["k90"]) / window_count(n_qubits, width),
                        "terminal_support": float(profile["terminal_support"]),
                        "labels_requested": len(labels),
                        "labels_used": int(profile["labels_used"]),
                        "capped_labels": int(profile["capped_labels"]),
                        "avg_terminal_terms": float(profile["avg_terminal_terms"]),
                        "all_pauli_log10_k90": all_row["log10_k90"],
                        "log10_k90_over_all_pauli": math.log10(max(float(profile["k90"]), 1e-300))
                        - float(all_row["log10_k90"]),
                        "k90_over_all_pauli": 10.0
                        ** (
                            math.log10(max(float(profile["k90"]), 1e-300))
                            - float(all_row["log10_k90"])
                        ),
                    }
                )
        rows.append(all_row)
    return rows


def run_estimator_stress() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(RNG_SEED + 4)
    n_qubits = 12
    width = 3
    gates = stress_channel(n_qubits, depth=3)
    cases = [
        ("window_z_string", window_z_string_labels(n_qubits, width)),
        ("window_x_string", window_x_string_labels(n_qubits, width)),
        ("window_xy_ensemble", window_xy_full_labels(n_qubits, width)),
        ("sampled_all_pauli", sample_all_labels(n_qubits, random.Random(RNG_SEED + 5), 72)),
    ]
    sample_counts = [128, 256, 512, 1024, 2048]
    trials = 180
    estimator_rows: list[dict[str, object]] = []
    truncation_rows: list[dict[str, object]] = []
    for task, labels in cases:
        terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=250_000)
        profile = terminal_truncation_profile(terminal)
        truncation_rows.extend(
            terminal_observable_truncation_errors(
                terminal,
                {
                    "case": task,
                    "n_qubits": n_qubits,
                    "layers": 3,
                    "t_density": "",
                },
                targets=[0.50, 0.70, 0.90, 0.95],
            )
        )
        p = float(terminal["collision_probability"])
        for samples in sample_counts:
            estimates = [
                run_terminal_collision_estimator(terminal["columns"], samples, rng)
                for _ in range(trials)
            ]
            estimate_array = np.asarray(estimates, dtype=float)
            rmse = float(np.sqrt(np.mean((estimate_array - p) ** 2)))
            relative_rmse = rmse / max(p, 1e-300)
            theory = math.sqrt((1.0 - p) / max(samples * p, 1e-300))
            estimator_rows.append(
                {
                    "case": task,
                    "n_qubits": n_qubits,
                    "window_width": width if task != "sampled_all_pauli" else math.nan,
                    "samples": samples,
                    "trials": trials,
                    "task_p2": profile["task_p2"],
                    "collision_probability": p,
                    "effective_samples": samples * p,
                    "relative_rmse": relative_rmse,
                    "theory_relative_rmse": theory,
                    "rmse_over_theory": relative_rmse / max(theory, 1e-300),
                    "k90": profile["k90"],
                    "labels_used": terminal["labels_used"],
                    "capped_labels": terminal["capped_labels"],
                    "avg_terminal_terms": terminal["avg_terminal_terms"],
                }
            )
    return estimator_rows, truncation_rows


def stress_channel(n_qubits: int, depth: int) -> list[Gate]:
    gates: list[Gate] = []
    for layer in range(depth):
        for q in range(n_qubits):
            if (q + layer) % 2 == 0:
                gates.append(gate("H", q))
            else:
                gates.append(gate("S", q))
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            gates.append(gate("CZ", q, q + 1))
        for q in range(n_qubits):
            gates.append(gate("T", q))
        for q in range(1 - start, n_qubits - 1, 2):
            gates.append(gate("CNOT", q, q + 1))
    return gates


def window_z_string_labels(n_qubits: int, width: int) -> list[Label]:
    return [(0, window_mask(start, width)) for start in range(window_count(n_qubits, width))]


def window_x_string_labels(n_qubits: int, width: int) -> list[Label]:
    return [(window_mask(start, width), 0) for start in range(window_count(n_qubits, width))]


def window_xy_full_labels(n_qubits: int, width: int) -> list[Label]:
    labels: list[Label] = []
    for start in range(window_count(n_qubits, width)):
        mask = window_mask(start, width)
        bits = [1 << q for q in range(start, start + width)]
        for z_pattern in range(1 << width):
            zmask = 0
            for offset, bit in enumerate(bits):
                if z_pattern & (1 << offset):
                    zmask |= bit
            labels.append((mask, zmask))
    return labels


def window_mask(start: int, width: int) -> int:
    mask = 0
    for q in range(start, start + width):
        mask |= 1 << q
    return mask


def window_count(n_qubits: int, width: int) -> int:
    if width < 1 or width > n_qubits:
        raise ValueError("window width must be between 1 and n_qubits")
    return n_qubits - width + 1


def all_pauli_row(n_qubits: int) -> dict[str, object]:
    log10_all = n_qubits * math.log10(4.0)
    log10_k90 = math.log10(0.9) + log10_all
    return {
        "family": "local_window_stress",
        "n_qubits": n_qubits,
        "window_width": math.nan,
        "task": "all_pauli",
        "task_description": "uniform all-Pauli endpoint, analytic",
        "operator_magic": math.nan,
        "task_p2": math.nan,
        "exp_task_p2": math.nan,
        "k90": 10.0**log10_k90,
        "log10_k90": log10_k90,
        "k90_per_window": math.nan,
        "terminal_support": 10.0**log10_all,
        "labels_requested": math.nan,
        "labels_used": math.nan,
        "capped_labels": 0,
        "avg_terminal_terms": math.nan,
        "all_pauli_log10_k90": log10_k90,
        "log10_k90_over_all_pauli": 0.0,
        "k90_over_all_pauli": 1.0,
    }


def summarize(
    workload_rows: list[dict[str, object]],
    estimator_rows: list[dict[str, object]],
    truncation_rows: list[dict[str, object]],
) -> dict[str, object]:
    local_rows = [row for row in workload_rows if row["task"] != "all_pauli"]
    largest_n = max(int(row["n_qubits"]) for row in local_rows)
    largest_rows = [row for row in local_rows if int(row["n_qubits"]) == largest_n]
    max_local_ratio = max(float(row["k90_over_all_pauli"]) for row in largest_rows)
    width5_rows = [
        row for row in largest_rows if int(row["window_width"]) == 5 and row["task"] != "all_pauli"
    ]
    rmse_ratios = [
        float(row["rmse_over_theory"])
        for row in estimator_rows
        if float(row["effective_samples"]) >= 1.0 and int(row["capped_labels"]) == 0
    ]
    useful_trunc = [
        row
        for row in truncation_rows
        if row["state_response"] == "tilted_product" and float(row["target_mass"]) >= 0.9
        and int(row["capped_labels"]) == 0
    ]
    return {
        "largest_n": largest_n,
        "max_selected_over_all_at_largest_n": max_local_ratio,
        "width5_k90_per_window": {
            str(row["task"]): float(row["k90_per_window"]) for row in width5_rows
        },
        "median_rmse_over_theory": float(np.median(rmse_ratios)) if rmse_ratios else math.nan,
        "max_rmse_over_theory": max(rmse_ratios) if rmse_ratios else math.nan,
        "tilted_eta90_max_relative_error": max(
            float(row["relative_rms_error"]) for row in useful_trunc
        )
        if useful_trunc
        else math.nan,
    }


def plot_stress(
    workload_rows: list[dict[str, object]],
    estimator_rows: list[dict[str, object]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.15), dpi=300)
    styles = {
        "window_z_string": ("o", "#111111", "local $Z$ string"),
        "window_x_string": ("s", "#4d4d4d", "local $X$ string"),
        "window_xy_ensemble": ("D", "#8a8a8a", "local $X/Y$ ensemble"),
    }
    n_plot = max(int(row["n_qubits"]) for row in workload_rows)
    for task, (marker, color, label) in styles.items():
        subset = sorted(
            [
                row
                for row in workload_rows
                if row["task"] == task and int(row["n_qubits"]) == n_plot
            ],
            key=lambda row: int(row["window_width"]),
        )
        axes[0].plot(
            [int(row["window_width"]) for row in subset],
            [float(row["k90_per_window"]) for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=color,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("local window size")
    axes[0].set_ylabel(r"$K_{0.9}$/window")
    axes[0].set_title(r"(a) Local support controls cost", fontsize=8.0, pad=2)

    width_plot = 3
    for task, (marker, color, label) in styles.items():
        subset = sorted(
            [
                row
                for row in workload_rows
                if row["task"] == task and int(row["window_width"]) == width_plot
            ],
            key=lambda row: int(row["n_qubits"]),
        )
        axes[1].plot(
            [int(row["n_qubits"]) for row in subset],
            [float(row["log10_k90"]) for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=color,
            label=label,
        )
    all_rows = sorted(
        [row for row in workload_rows if row["task"] == "all_pauli"],
        key=lambda row: int(row["n_qubits"]),
    )
    axes[1].plot(
        [int(row["n_qubits"]) for row in all_rows],
        [float(row["log10_k90"]) for row in all_rows],
        marker="^",
        markersize=3.0,
        linewidth=0.9,
        linestyle="--",
        color="#111111",
        markerfacecolor="white",
        label="all Pauli",
    )
    axes[1].set_xlabel("qubits")
    axes[1].set_ylabel(r"$\log_{10}K_{0.9}$")
    axes[1].set_title(r"(b) Same channel, fixed local window", fontsize=8.0, pad=2)

    valid = [
        row
        for row in estimator_rows
        if float(row["effective_samples"]) > 0 and int(row["capped_labels"]) == 0
    ]
    axes[2].scatter(
        [1.0 / math.sqrt(float(row["effective_samples"])) for row in valid],
        [float(row["relative_rmse"]) for row in valid],
        s=11,
        color="#4d4d4d",
        alpha=0.82,
    )
    xs = np.linspace(
        min(1.0 / math.sqrt(float(row["effective_samples"])) for row in valid),
        max(1.0 / math.sqrt(float(row["effective_samples"])) for row in valid),
        100,
    )
    axes[2].plot(xs, xs, color="#111111", linewidth=0.8, linestyle="--", label="unit slope")
    axes[2].set_xlabel(r"$(m e^{-\mathcal{P}_2})^{-1/2}$")
    axes[2].set_ylabel("relative RMSE")
    axes[2].set_title(r"(c) Collision estimator collapse", fontsize=8.0, pad=2)
    for ax in axes:
        ax.tick_params(axis="both", length=3)
        ax.grid(True, linewidth=0.3, alpha=0.22)
    handles, labels = collect_legend(axes)
    fig.tight_layout(rect=(0, 0.16, 1, 1), pad=0.2, w_pad=0.35)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
    )
    save_figure(fig, "fig_stress_workloads")


def collect_legend(axes: Iterable[object]) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    return handles, labels


def save_figure(fig: object, stem: str) -> None:
    fig.savefig(RESULTS_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def task_description(task: str) -> str:
    return {
        "window_z_string": "contiguous local Z string averaged over positions",
        "window_x_string": "contiguous local X string averaged over positions",
        "window_xy_ensemble": "all transverse X/Y strings inside each local window",
    }[task]


def label_to_index(label: Label) -> int:
    xmask, zmask = label
    nbits = max(xmask.bit_length(), zmask.bit_length())
    index = 0
    for q in range(nbits):
        x_bit = bool(xmask & (1 << q))
        z_bit = bool(zmask & (1 << q))
        if x_bit and z_bit:
            code = 2
        elif x_bit:
            code = 1
        elif z_bit:
            code = 3
        else:
            code = 0
        index += code * (4**q)
    return index


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            converted: dict[str, object] = {}
            for key, value in row.items():
                converted[key] = convert_value(value)
            rows.append(converted)
    return rows


def convert_value(value: str) -> object:
    if value == "":
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
