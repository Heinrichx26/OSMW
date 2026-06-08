from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r41_observable_cost import (
    Label,
    build_terminal_columns,
    exact_sparse_task_p2,
    terminal_truncation_profile,
)
from r41_path_entropy import Gate, gate, pauli_transfer_matrix, unitary_from_gates
from r41_task_resolved_checks import task_entropy_from_transfer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = read_csv(DATA_DIR / "r41_2d_stress_workloads.csv")
        plot_2d_stress(rows)
        return

    validation = run_quick_validation()
    (DATA_DIR / "r41_2d_stress_workloads_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.quick:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    rows = run_2d_stress()
    write_csv(DATA_DIR / "r41_2d_stress_workloads.csv", rows)
    summary = summarize(rows)
    (DATA_DIR / "r41_2d_stress_workloads.json").write_text(
        json.dumps(
            {
                "validation": validation,
                "summary": summary,
                "workload_rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_2d_stress(rows)


def run_quick_validation() -> dict[str, object]:
    exact_lattice = 2
    exact_n = exact_lattice * exact_lattice
    exact_gates = square_lattice_channel(exact_lattice, depth=1)
    labels = plaquette_xy_labels(exact_lattice)
    transfer = pauli_transfer_matrix(unitary_from_gates(exact_n, exact_gates))
    exact = task_entropy_from_transfer(
        transfer,
        [label_to_index(label) for label in labels],
    )
    sparse = exact_sparse_task_p2(exact_n, exact_gates, labels, max_terms=40_000)

    quick_lattice = 3
    quick_n = quick_lattice * quick_lattice
    quick_gates = square_lattice_channel(quick_lattice, depth=1)
    quick_labels = plaquette_xy_labels(quick_lattice)
    terminal = build_terminal_columns(quick_n, quick_gates, quick_labels, max_terms=80_000)
    profile = terminal_truncation_profile(terminal)
    plaquette_ratio = float(profile["k90"]) / math.ceil(0.9 * (4**quick_n))
    checks = [
        {
            "case": "exact_ptm_vs_sparse_l2_plaquette_xy",
            "observed": sparse,
            "expected": exact,
            "abs_error": abs(sparse - exact),
            "passed": abs(sparse - exact) < 1e-10,
        },
        {
            "case": "l3_plaquette_xy_below_all_pauli",
            "observed_ratio": plaquette_ratio,
            "passed": plaquette_ratio < 1.0,
        },
        {
            "case": "l3_no_sparse_cap",
            "capped_labels": int(terminal["capped_labels"]),
            "passed": int(terminal["capped_labels"]) == 0,
        },
    ]
    return {
        "case": "two_dimensional_local_window_quick_validation",
        "checks": checks,
        "max_abs_error": max(float(item.get("abs_error", 0.0)) for item in checks),
        "plaquette_ratio": plaquette_ratio,
        "passed": all(bool(item["passed"]) for item in checks),
    }


def run_2d_stress(
    lattice_values: list[int] | None = None,
    depth: int = 2,
    max_terms: int = 250_000,
) -> list[dict[str, object]]:
    if lattice_values is None:
        lattice_values = [3, 4, 5, 6]
    rows: list[dict[str, object]] = []
    for lattice in lattice_values:
        n_qubits = lattice * lattice
        gates = square_lattice_channel(lattice, depth=depth)
        all_row = all_pauli_row(lattice)
        specs = [
            ("site_z", 1, site_z_labels(lattice), site_count(lattice)),
            ("site_xy", 1, site_xy_labels(lattice), site_count(lattice)),
            ("bond_xx", 2, bond_xx_labels(lattice), bond_count(lattice)),
            ("plaquette_xy", 4, plaquette_xy_labels(lattice), plaquette_count(lattice)),
        ]
        for task, support_size, labels, units in specs:
            terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=max_terms)
            profile = terminal_truncation_profile(terminal)
            k90 = float(profile["k90"])
            log10_k90 = math.log10(max(k90, 1e-300))
            all_log10 = float(all_row["log10_k90"])
            rows.append(
                {
                    "family": "two_dimensional_local_window_stress",
                    "lattice_L": lattice,
                    "n_qubits": n_qubits,
                    "depth": depth,
                    "task": task,
                    "task_description": task_description(task),
                    "support_size": support_size,
                    "local_units": units,
                    "task_p2": profile["task_p2"],
                    "exp_task_p2": profile["exp_task_p2"],
                    "k90": k90,
                    "log10_k90": log10_k90,
                    "k90_per_local_unit": k90 / units,
                    "terminal_support": float(profile["terminal_support"]),
                    "labels_requested": len(labels),
                    "labels_used": int(profile["labels_used"]),
                    "capped_labels": int(profile["capped_labels"]),
                    "avg_terminal_terms": float(profile["avg_terminal_terms"]),
                    "all_pauli_log10_k90": all_log10,
                    "log10_k90_over_all_pauli": log10_k90 - all_log10,
                    "k90_over_all_pauli": 10.0 ** (log10_k90 - all_log10),
                }
            )
        rows.append(all_row)
    return rows


def square_lattice_channel(lattice: int, depth: int) -> list[Gate]:
    gates: list[Gate] = []
    for layer in range(depth):
        for q in range(lattice * lattice):
            gates.append(gate("H" if (q + layer) % 2 == 0 else "S", q))
        gates.extend(square_cz_edges(lattice, parity=layer % 2))
        for q in range(lattice * lattice):
            gates.append(gate("T", q))
        gates.extend(square_cnot_edges(lattice, parity=1 - (layer % 2)))
    return gates


def square_cz_edges(lattice: int, parity: int) -> list[Gate]:
    gates: list[Gate] = []
    for r in range(lattice):
        for c in range(lattice - 1):
            if (r + c) % 2 == parity:
                gates.append(gate("CZ", site_index(lattice, r, c), site_index(lattice, r, c + 1)))
    for r in range(lattice - 1):
        for c in range(lattice):
            if (r + c) % 2 == parity:
                gates.append(gate("CZ", site_index(lattice, r, c), site_index(lattice, r + 1, c)))
    return gates


def square_cnot_edges(lattice: int, parity: int) -> list[Gate]:
    gates: list[Gate] = []
    for r in range(lattice):
        for c in range(lattice - 1):
            if (r + c) % 2 == parity:
                gates.append(gate("CNOT", site_index(lattice, r, c), site_index(lattice, r, c + 1)))
    for r in range(lattice - 1):
        for c in range(lattice):
            if (r + c) % 2 == parity:
                gates.append(gate("CNOT", site_index(lattice, r, c), site_index(lattice, r + 1, c)))
    return gates


def site_z_labels(lattice: int) -> list[Label]:
    return [(0, 1 << q) for q in range(lattice * lattice)]


def site_xy_labels(lattice: int) -> list[Label]:
    labels: list[Label] = []
    for q in range(lattice * lattice):
        bit = 1 << q
        labels.append((bit, 0))
        labels.append((bit, bit))
    return labels


def bond_xx_labels(lattice: int) -> list[Label]:
    labels: list[Label] = []
    for q0, q1 in nearest_neighbor_edges(lattice):
        labels.append(((1 << q0) | (1 << q1), 0))
    return labels


def plaquette_xy_labels(lattice: int) -> list[Label]:
    labels: list[Label] = []
    for r in range(lattice - 1):
        for c in range(lattice - 1):
            sites = [
                site_index(lattice, r, c),
                site_index(lattice, r, c + 1),
                site_index(lattice, r + 1, c),
                site_index(lattice, r + 1, c + 1),
            ]
            xmask = sum(1 << q for q in sites)
            for z_pattern in range(1 << len(sites)):
                zmask = 0
                for offset, q in enumerate(sites):
                    if z_pattern & (1 << offset):
                        zmask |= 1 << q
                labels.append((xmask, zmask))
    return labels


def nearest_neighbor_edges(lattice: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(lattice):
        for c in range(lattice - 1):
            edges.append((site_index(lattice, r, c), site_index(lattice, r, c + 1)))
    for r in range(lattice - 1):
        for c in range(lattice):
            edges.append((site_index(lattice, r, c), site_index(lattice, r + 1, c)))
    return edges


def site_index(lattice: int, row: int, col: int) -> int:
    return row * lattice + col


def site_count(lattice: int) -> int:
    return lattice * lattice


def bond_count(lattice: int) -> int:
    return 2 * lattice * (lattice - 1)


def plaquette_count(lattice: int) -> int:
    return (lattice - 1) * (lattice - 1)


def all_pauli_row(lattice: int) -> dict[str, object]:
    n_qubits = lattice * lattice
    log10_all = n_qubits * math.log10(4.0)
    log10_k90 = math.log10(0.9) + log10_all
    return {
        "family": "two_dimensional_local_window_stress",
        "lattice_L": lattice,
        "n_qubits": n_qubits,
        "depth": math.nan,
        "task": "all_pauli",
        "task_description": "uniform all-Pauli endpoint, analytic",
        "support_size": math.nan,
        "local_units": math.nan,
        "task_p2": math.nan,
        "exp_task_p2": math.nan,
        "k90": 10.0**log10_k90,
        "log10_k90": log10_k90,
        "k90_per_local_unit": math.nan,
        "terminal_support": 10.0**log10_all,
        "labels_requested": math.nan,
        "labels_used": math.nan,
        "capped_labels": 0,
        "avg_terminal_terms": math.nan,
        "all_pauli_log10_k90": log10_k90,
        "log10_k90_over_all_pauli": 0.0,
        "k90_over_all_pauli": 1.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if row["task"] != "all_pauli"]
    largest_l = max(int(row["lattice_L"]) for row in selected)
    largest = [row for row in selected if int(row["lattice_L"]) == largest_l]
    plaquette = [row for row in largest if row["task"] == "plaquette_xy"][0]
    capped_total = sum(int(row["capped_labels"]) for row in selected)
    return {
        "largest_lattice_L": largest_l,
        "largest_n": largest_l * largest_l,
        "max_selected_over_all_at_largest_lattice": max(
            float(row["k90_over_all_pauli"]) for row in largest
        ),
        "plaquette_xy_k90_per_plaquette": float(plaquette["k90_per_local_unit"]),
        "plaquette_xy_selected_over_all": float(plaquette["k90_over_all_pauli"]),
        "capped_labels_total": capped_total,
    }


def plot_2d_stress(rows: list[dict[str, object]]) -> None:
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
        "site_z": ("o", "#111111", "site $Z$"),
        "site_xy": ("s", "#4d4d4d", "site $X/Y$"),
        "bond_xx": ("^", "#6a6a6a", "bond $XX$"),
        "plaquette_xy": ("D", "#999999", "plaquette $X/Y$"),
    }
    for task, (marker, color, label) in styles.items():
        subset = sorted(
            [row for row in rows if row["task"] == task],
            key=lambda row: int(row["lattice_L"]),
        )
        axes[0].plot(
            [int(row["lattice_L"]) for row in subset],
            [float(row["k90_per_local_unit"]) for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=color,
            label=label,
        )
        axes[1].plot(
            [int(row["n_qubits"]) for row in subset],
            [float(row["log10_k90"]) for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=color,
        )
        axes[2].plot(
            [int(row["n_qubits"]) for row in subset],
            [float(row["k90_over_all_pauli"]) for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            color=color,
        )
    all_rows = sorted(
        [row for row in rows if row["task"] == "all_pauli"],
        key=lambda row: int(row["n_qubits"]),
    )
    axes[1].plot(
        [int(row["n_qubits"]) for row in all_rows],
        [float(row["log10_k90"]) for row in all_rows],
        marker="v",
        markersize=3.0,
        linewidth=0.9,
        linestyle="--",
        color="#111111",
        markerfacecolor="white",
        label="all Pauli",
    )
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_xlabel("linear size")
    axes[0].set_ylabel(r"$K_{0.9}$/local unit")
    axes[0].set_title("(a) Local 2D windows", fontsize=8.0, pad=2)
    axes[1].set_xlabel("qubits")
    axes[1].set_ylabel(r"$\log_{10}K_{0.9}$")
    axes[1].set_title("(b) Same 2D channels", fontsize=8.0, pad=2)
    axes[2].set_xlabel("qubits")
    axes[2].set_ylabel(r"$K_{0.9}/K_{0.9}({\rm all})$")
    axes[2].set_title("(c) Selected/all ratio", fontsize=8.0, pad=2)
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
        ncol=5,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.65,
    )
    fig.savefig(RESULTS_DIR / "fig_2d_stress_workloads.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig_2d_stress_workloads.png", bbox_inches="tight")
    plt.close(fig)


def task_description(task: str) -> str:
    return {
        "site_z": "single-site Z probes on a square lattice",
        "site_xy": "single-site X/Y probes on a square lattice",
        "bond_xx": "nearest-neighbor XX bond terms on a square lattice",
        "plaquette_xy": "all transverse X/Y strings on each square plaquette",
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
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: convert_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def convert_value(value: str) -> object:
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


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


if __name__ == "__main__":
    main()
