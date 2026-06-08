from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

from r41_path_entropy import pauli_transfer_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate observable-conditioned actions for Rz(theta)^n."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a single small validation case before the full grid",
    )
    args = parser.parse_args()
    quick = args.smoke or args.quick

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if quick:
        rows = _run_grid(n_values=[1], theta_values=[("pi/6", math.pi / 6.0)])
        rows.extend(_local_action_rows([("pi/6", math.pi / 6.0)]))
        validation = _validation_payload(rows, smoke=True)
        _write_json(DATA_DIR / "r41_rotation_action_quick.json", validation)
        print(json.dumps(validation, indent=2))
        return

    theta_values = [
        ("pi/8", math.pi / 8.0),
        ("pi/6", math.pi / 6.0),
        ("pi/4", math.pi / 4.0),
    ]
    rows = _run_grid(n_values=[1, 2, 3, 4], theta_values=theta_values)
    rows.extend(_local_action_rows(theta_values))
    _write_csv(DATA_DIR / "r41_rotation_action_checks.csv", rows)
    _write_json(DATA_DIR / "r41_rotation_action_checks.json", {"rows": rows})
    validation = _validation_payload(rows, smoke=False)
    _write_json(DATA_DIR / "r41_rotation_action_validation.json", validation)
    print(json.dumps(validation, indent=2))


def _run_grid(
    n_values: list[int],
    theta_values: list[tuple[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_qubits in n_values:
        labels = _pauli_labels(n_qubits)
        tasks = _task_indices(labels)
        for theta_label, theta in theta_values:
            unitary = _rz_tensor_unitary(n_qubits, theta)
            transfer = pauli_transfer_matrix(unitary)
            for task, indices in tasks.items():
                ptm_p2 = _task_p2(transfer, indices)
                analytic_p2, formula = _analytic_p2(n_qubits, theta, task)
                rows.append(
                    {
                        "family": "Rz_tensor",
                        "n_qubits": n_qubits,
                        "theta_label": theta_label,
                        "theta": theta,
                        "task": task,
                        "omega_size": len(indices),
                        "ptm_p2": ptm_p2,
                        "analytic_p2": analytic_p2,
                        "abs_error": abs(ptm_p2 - analytic_p2),
                        "formula": formula,
                    }
                )
    return rows


def _local_action_rows(theta_values: list[tuple[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for theta_label, theta in theta_values:
        numeric_alpha = _numeric_alpha4_rotation(theta)
        analytic_alpha = _analytic_alpha4_rotation(theta)
        rows.append(
            {
                "family": "Rz_local_action",
                "n_qubits": 1,
                "theta_label": theta_label,
                "theta": theta,
                "task": "worst_case_transverse",
                "omega_size": 2,
                "ptm_p2": -math.log(max(numeric_alpha, 1e-300)),
                "analytic_p2": -math.log(max(analytic_alpha, 1e-300)),
                "abs_error": abs(numeric_alpha - analytic_alpha),
                "formula": "-log alpha4, alpha4=((9-cos(4 theta))-sqrt((9-cos(4 theta))^2-64))/8",
            }
        )
    return rows


def _rz_tensor_unitary(n_qubits: int, theta: float) -> np.ndarray:
    dim = 1 << n_qubits
    diagonal = np.zeros(dim, dtype=np.complex128)
    for basis in range(dim):
        weight = basis.bit_count()
        z_sum = n_qubits - 2 * weight
        diagonal[basis] = np.exp(-0.5j * theta * z_sum)
    return np.diag(diagonal)


def _pauli_labels(n_qubits: int) -> list[tuple[int, ...]]:
    return list(product(range(4), repeat=n_qubits))


def _task_indices(labels: list[tuple[int, ...]]) -> dict[str, list[int]]:
    n_qubits = len(labels[0])
    predicates: dict[str, Callable[[tuple[int, ...]], bool]] = {
        "local_z": lambda codes: codes[0] == 3 and all(code == 0 for code in codes[1:]),
        "local_transverse": lambda codes: codes[0] in {1, 2}
        and all(code == 0 for code in codes[1:]),
        "full_transverse": lambda codes: all(code in {1, 2} for code in codes),
        "all_pauli": lambda codes: True,
    }
    return {
        task: [index for index, codes in enumerate(labels) if predicate(codes)]
        for task, predicate in predicates.items()
    }


def _task_p2(transfer: np.ndarray, indices: list[int]) -> float:
    if not indices:
        raise ValueError("task index set must be nonempty")
    collision = float(np.mean([np.sum(transfer[:, index] ** 4) for index in indices]))
    value = -math.log(max(collision, 1e-300))
    return 0.0 if abs(value) < 1e-12 else value


def _analytic_p2(n_qubits: int, theta: float, task: str) -> tuple[float, str]:
    gamma = math.cos(theta) ** 4 + math.sin(theta) ** 4
    branch_action = -math.log(gamma)
    if task == "local_z":
        return 0.0, "0"
    if task == "local_transverse":
        return branch_action, "-log(cos(theta)^4+sin(theta)^4)"
    if task == "full_transverse":
        return n_qubits * branch_action, "-n log(cos(theta)^4+sin(theta)^4)"
    if task == "all_pauli":
        return -n_qubits * math.log((1.0 + gamma) / 2.0), "-n log((1+gamma)/2)"
    raise ValueError(f"unknown task: {task}")


def _analytic_alpha4_rotation(theta: float) -> float:
    c = math.cos(4.0 * theta)
    gap = 9.0 - c
    return (gap - math.sqrt(max(gap * gap - 64.0, 0.0))) / 8.0


def _numeric_alpha4_rotation(theta: float) -> float:
    def ratio(phi: float) -> float:
        numerator = math.cos(phi + theta) ** 4 + math.sin(phi + theta) ** 4
        denominator = math.cos(phi) ** 4 + math.sin(phi) ** 4
        return numerator / denominator

    period = math.pi / 2.0
    samples = 512
    points = [period * index / samples for index in range(samples)]
    start = min(points, key=ratio)
    step = period / samples
    left = start - step
    right = start + step
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - gr * (right - left)
    x2 = left + gr * (right - left)
    f1 = ratio(x1)
    f2 = ratio(x2)
    for _ in range(120):
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + gr * (right - left)
            f2 = ratio(x2)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - gr * (right - left)
            f1 = ratio(x1)
    return min(f1, f2)


def _validation_payload(rows: list[dict[str, object]], smoke: bool) -> dict[str, object]:
    max_abs_error = max(float(row["abs_error"]) for row in rows)
    return {
        "status": "passed" if max_abs_error < 1e-12 else "failed",
        "quick": smoke,
        "num_rows": len(rows),
        "max_abs_error": max_abs_error,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
