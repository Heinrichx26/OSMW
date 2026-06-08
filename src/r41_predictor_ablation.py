from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from r41_observable_cost import (
    Label,
    hamiltonian_local_labels,
    local_transverse_labels,
    local_z_labels,
    random_clifford_t_circuit,
)
from r41_path_entropy import Gate, gate
from r41_pauli_scaling import _apply_clifford_label


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TASKS = {"local_z", "local_transverse", "hamiltonian_terms", "all_pauli"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_active_t_count()
    (DATA_DIR / "r41_predictor_ablation_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.validate:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    source_rows = read_global_task_rows(DATA_DIR / "r41_global_task_truncation.csv")
    point_rows = build_predictor_points(source_rows)
    fit_rows = fit_predictors(point_rows, "log_k90")
    write_csv(DATA_DIR / "r41_predictor_ablation_points.csv", point_rows)
    write_csv(DATA_DIR / "r41_predictor_ablation.csv", fit_rows)
    print(json.dumps({"fits": fit_rows}, indent=2))


def validate_active_t_count() -> dict[str, object]:
    n_qubits = 5
    t_layer = [gate("T", q) for q in range(n_qubits)]
    checks = [
        {
            "case": "local_z",
            "observed": average_active_t_count(n_qubits, t_layer, local_z_labels(n_qubits)),
            "expected": 0.0,
        },
        {
            "case": "local_transverse",
            "observed": average_active_t_count(
                n_qubits,
                t_layer,
                local_transverse_labels(n_qubits),
            ),
            "expected": 1.0,
        },
        {
            "case": "hamiltonian_terms",
            "observed": average_active_t_count(
                n_qubits,
                t_layer,
                hamiltonian_local_labels(n_qubits),
            ),
            "expected": 4.0 / 3.0,
        },
    ]
    max_error = max(abs(item["observed"] - item["expected"]) for item in checks)
    return {"checks": checks, "max_abs_error": max_error, "passed": max_error < 1e-12}


def read_global_task_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "layers": int(row["layers"]),
                    "t_density": float(row["t_density"]),
                    "expected_t_gates": float(row["expected_t_gates"]),
                    "repeat": int(row["repeat"]),
                    "circuit_seed": int(row["circuit_seed"]),
                    "global_p2_estimate": float(row["global_p2_estimate"]),
                    "task_p2": float(row["task_p2"]),
                    "k90": int(row["k90"]),
                }
            )
    return rows


def build_predictor_points(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    gate_cache: dict[tuple[int, int, float, int], list[Gate]] = {}
    active_cache: dict[tuple[int, int, float, int, str], float] = {}
    for row in rows:
        task = str(row["task"])
        if task not in TASKS or row["k90"] <= 0:
            continue
        key = (
            int(row["n_qubits"]),
            int(row["layers"]),
            float(row["t_density"]),
            int(row["circuit_seed"]),
        )
        if key not in gate_cache:
            gate_cache[key] = random_clifford_t_circuit(
                key[0],
                key[1],
                key[2],
                random.Random(key[3]),
            )
        gates = gate_cache[key]
        t_count = sum(1 for item in gates if item.name in {"T", "TDG"})
        active_key = (*key, task)
        if active_key not in active_cache:
            active_cache[active_key] = task_active_t_count(key[0], gates, task, t_count)
        output.append(
            {
                "n_qubits": row["n_qubits"],
                "layers": row["layers"],
                "t_density": row["t_density"],
                "repeat": row["repeat"],
                "circuit_seed": row["circuit_seed"],
                "task": task,
                "log_k90": math.log(float(row["k90"])),
                "k90": row["k90"],
                "global_operator_sre": row["global_p2_estimate"],
                "t_count": t_count,
                "active_t_count": active_cache[active_key],
                "task_p2": row["task_p2"],
            }
        )
    return output


def task_active_t_count(
    n_qubits: int,
    gates: list[Gate],
    task: str,
    total_t_count: int,
) -> float:
    if task == "all_pauli":
        return float(total_t_count)
    if task == "local_z":
        labels = local_z_labels(n_qubits)
    elif task == "local_transverse":
        labels = local_transverse_labels(n_qubits)
    elif task == "hamiltonian_terms":
        labels = hamiltonian_local_labels(n_qubits)
    else:
        raise ValueError(f"unsupported task: {task}")
    return average_active_t_count(n_qubits, gates, labels)


def average_active_t_count(n_qubits: int, gates: list[Gate], labels: list[Label]) -> float:
    if not labels:
        return 0.0
    return float(np.mean([active_t_count_for_label(n_qubits, gates, label) for label in labels]))


def active_t_count_for_label(n_qubits: int, gates: list[Gate], label: Label) -> int:
    branches: set[Label] = {label}
    active_indices: set[int] = set()
    for index, item in reversed(list(enumerate(gates))):
        next_branches: set[Label] = set()
        if item.name in {"T", "TDG"}:
            q = item.qubits[0]
            bit = 1 << q
            for xmask, zmask in branches:
                if xmask & bit:
                    active_indices.add(index)
                    next_branches.add((xmask, zmask & ~bit))
                    next_branches.add((xmask, zmask | bit))
                else:
                    next_branches.add((xmask, zmask))
        else:
            for branch in branches:
                next_branches.add(_apply_clifford_label(n_qubits, branch, item))
        branches = next_branches
    return len(active_indices)


def fit_predictors(rows: list[dict[str, object]], target: str) -> list[dict[str, object]]:
    y = [float(row[target]) for row in rows]
    output: list[dict[str, object]] = []
    for predictor, field, label in [
        ("global_operator_sre", "global_operator_sre", r"global \(M_2^{\rm op}\)"),
        ("t_count", "t_count", r"\(T\)-count"),
        ("active_t_count", "active_t_count", r"active \(T\) count"),
        ("task_p2", "task_p2", r"\(\mathcal P_2(U;O)\)"),
    ]:
        x = [float(row[field]) for row in rows]
        fit = linear_fit(x, y)
        output.append(
            {
                "target": target,
                "predictor": predictor,
                "label": label,
                "pairs": len(rows),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "coefficient_of_determination": fit["r2"],
                "pearson_correlation": fit["pearson"],
            }
        )
    return output


def linear_fit(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    design = np.vstack([np.ones(len(x)), x]).T
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 else 0.0
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "pearson": pearson,
    }


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


if __name__ == "__main__":
    main()
