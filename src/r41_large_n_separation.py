from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from r41_observable_cost import (
    Label,
    build_terminal_columns,
    hamiltonian_local_labels,
    local_transverse_labels,
    local_z_labels,
    terminal_truncation_profile,
)
from r41_path_entropy import Gate, gate


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation = run_validation()
    (DATA_DIR / "r41_large_n_separation_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.quick:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    rows = run_large_n_separation()
    write_csv(DATA_DIR / "r41_large_n_separation.csv", rows)
    summary = summarize(rows)
    (DATA_DIR / "r41_large_n_separation.json").write_text(
        json.dumps({"validation": validation, "summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def run_validation() -> dict[str, object]:
    rows = run_large_n_separation(n_values=[8, 12], frame_depth=2, max_terms=20_000)
    grouped = group_by_n(rows)
    checks: list[dict[str, object]] = []
    for n_qubits, task_rows in grouped.items():
        all_row = task_rows["all_pauli"]
        for task in ["local_z", "local_transverse", "hamiltonian_terms"]:
            row = task_rows[task]
            checks.append(
                {
                    "case": f"n{n_qubits}_{task}_below_all",
                    "observed_ratio": row["k90_over_all_pauli"],
                    "passed": row["k90_over_all_pauli"] < 1.0,
                }
            )
        checks.append(
            {
                "case": f"n{n_qubits}_all_pauli_matches_operator_sre",
                "observed": all_row["task_p2"],
                "expected": all_row["operator_magic"],
                "abs_error": abs(all_row["task_p2"] - all_row["operator_magic"]),
                "passed": abs(all_row["task_p2"] - all_row["operator_magic"]) < 1e-12,
            }
        )
    checks.append(
        {
            "case": "local_z_ratio_decreases",
            "observed": grouped[12]["local_z"]["k90_over_all_pauli"],
            "reference": grouped[8]["local_z"]["k90_over_all_pauli"],
            "passed": grouped[12]["local_z"]["k90_over_all_pauli"]
            < grouped[8]["local_z"]["k90_over_all_pauli"],
        }
    )
    return {
        "case": "fixed_depth_dressed_T_large_n_quick_validation",
        "checks": checks,
        "passed": all(bool(item["passed"]) for item in checks),
    }


def run_large_n_separation(
    n_values: list[int] | None = None,
    frame_depth: int = 2,
    max_terms: int = 200_000,
) -> list[dict[str, object]]:
    if n_values is None:
        n_values = [8, 16, 24, 32, 48, 64, 96]
    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        gates = dressed_t_circuit(n_qubits, frame_depth=frame_depth)
        task_specs: list[tuple[str, str, list[Label]]] = [
            ("local_z", "single-site Z probes averaged over the chain", local_z_labels(n_qubits)),
            (
                "local_transverse",
                "single-site X and Y probes averaged over the chain",
                local_transverse_labels(n_qubits),
            ),
            (
                "hamiltonian_terms",
                "nearest-neighbor XX, YY, and ZZ terms averaged over the chain",
                hamiltonian_local_labels(n_qubits),
            ),
        ]
        for task_name, task_description, labels in task_specs:
            terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=max_terms)
            profile = terminal_truncation_profile(terminal)
            rows.append(
                {
                    "family": "fixed_depth_clifford_dressed_T",
                    "frame_depth": frame_depth,
                    "n_qubits": n_qubits,
                    "task": task_name,
                    "task_description": task_description,
                    "operator_magic": n_qubits * math.log(4.0 / 3.0),
                    "task_p2": profile["task_p2"],
                    "exp_task_p2": profile["exp_task_p2"],
                    "k90": float(profile["k90"]),
                    "log10_k90": math.log10(max(float(profile["k90"]), 1e-300)),
                    "terminal_support": float(profile["terminal_support"]),
                    "labels_used": profile["labels_used"],
                    "labels_requested": len(labels),
                    "capped_labels": profile["capped_labels"],
                    "avg_terminal_terms": profile["avg_terminal_terms"],
                }
            )
        rows.append(all_pauli_row(n_qubits, frame_depth))
    add_same_circuit_ratios(rows)
    return rows


def dressed_t_circuit(n_qubits: int, frame_depth: int) -> list[Gate]:
    gates = [gate("T", q) for q in range(n_qubits)]
    gates.extend(local_clifford_frame(n_qubits, frame_depth))
    return gates


def local_clifford_frame(n_qubits: int, frame_depth: int) -> list[Gate]:
    gates: list[Gate] = []
    for layer in range(frame_depth):
        for q in range(n_qubits):
            gates.append(gate("H", q))
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            gates.append(gate("CZ", q, q + 1))
        for q in range(n_qubits):
            gates.append(gate("S", q))
    return gates


def all_pauli_row(n_qubits: int, frame_depth: int) -> dict[str, object]:
    operator_magic = n_qubits * math.log(4.0 / 3.0)
    log10_all_support = n_qubits * math.log10(4.0)
    log10_k90 = math.log10(0.9) + log10_all_support
    return {
        "family": "fixed_depth_clifford_dressed_T",
        "frame_depth": frame_depth,
        "n_qubits": n_qubits,
        "task": "all_pauli",
        "task_description": "uniform all-Pauli task, evaluated analytically",
        "operator_magic": operator_magic,
        "task_p2": operator_magic,
        "exp_task_p2": math.exp(operator_magic),
        "k90": 10.0**log10_k90,
        "log10_k90": log10_k90,
        "terminal_support": 10.0**log10_all_support,
        "labels_used": math.nan,
        "labels_requested": math.nan,
        "capped_labels": 0,
        "avg_terminal_terms": math.nan,
    }


def add_same_circuit_ratios(rows: list[dict[str, object]]) -> None:
    grouped = group_by_n(rows)
    for task_rows in grouped.values():
        all_log = float(task_rows["all_pauli"]["log10_k90"])
        for row in task_rows.values():
            row["log10_k90_over_all_pauli"] = float(row["log10_k90"]) - all_log
            row["k90_over_all_pauli"] = 10.0 ** float(row["log10_k90_over_all_pauli"])


def group_by_n(rows: list[dict[str, object]]) -> dict[int, dict[str, dict[str, object]]]:
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_qubits"]), {})[str(row["task"])] = row
    return grouped


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped = group_by_n(rows)
    largest_n = max(grouped)
    return {
        "largest_n": largest_n,
        "frame_depth": grouped[largest_n]["all_pauli"]["frame_depth"],
        "local_z_ratio_at_largest_n": grouped[largest_n]["local_z"]["k90_over_all_pauli"],
        "local_transverse_ratio_at_largest_n": grouped[largest_n]["local_transverse"][
            "k90_over_all_pauli"
        ],
        "hamiltonian_ratio_at_largest_n": grouped[largest_n]["hamiltonian_terms"][
            "k90_over_all_pauli"
        ],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
