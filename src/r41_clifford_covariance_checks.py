from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from r41_path_entropy import gate, pauli_transfer_matrix, unitary_from_gates


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Clifford covariance of task-resolved quantities."
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help="run one small check")
    args = parser.parse_args()
    quick = args.smoke or args.quick

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = _run_checks(smoke=quick)
    validation = _validation_payload(rows, smoke=quick)
    if quick:
        _write_json(DATA_DIR / "r41_clifford_covariance_quick.json", validation)
    else:
        _write_csv(DATA_DIR / "r41_clifford_covariance_checks.csv", rows)
        _write_json(DATA_DIR / "r41_clifford_covariance_checks.json", {"rows": rows})
        _write_json(DATA_DIR / "r41_clifford_covariance_validation.json", validation)
    print(json.dumps(validation, indent=2))


def _run_checks(smoke: bool) -> list[dict[str, object]]:
    n_qubits = 2 if smoke else 3
    u_gates = (
        [gate("T", 0), gate("CNOT", 0, 1), gate("T", 1)]
        if smoke
        else [
            gate("T", 0),
            gate("CNOT", 0, 1),
            gate("H", 2),
            gate("T", 2),
            gate("CZ", 1, 2),
            gate("T", 1),
        ]
    )
    cin_gates = (
        [gate("H", 0), gate("S", 1)]
        if smoke
        else [gate("H", 0), gate("S", 1), gate("CNOT", 1, 2), gate("H", 2)]
    )
    cout_gates = (
        [gate("CNOT", 0, 1), gate("H", 1)]
        if smoke
        else [gate("CZ", 0, 1), gate("H", 1), gate("S", 2), gate("CNOT", 0, 2)]
    )
    eta = 0.86 if smoke else 0.91

    unitary_u = unitary_from_gates(n_qubits, u_gates)
    unitary_cin = unitary_from_gates(n_qubits, cin_gates)
    unitary_cout = unitary_from_gates(n_qubits, cout_gates)
    unitary_v = unitary_cout @ unitary_u @ unitary_cin

    transfer_u = pauli_transfer_matrix(unitary_u)
    transfer_v = pauli_transfer_matrix(unitary_v)
    transfer_cin = pauli_transfer_matrix(unitary_cin)
    transfer_cout = pauli_transfer_matrix(unitary_cout)

    labels = _pauli_labels(n_qubits)
    weights_v = _task_weights(labels, smoke=smoke)
    weights_u = (transfer_cout**2) @ weights_v

    collision_v = _task_collision(transfer_v, weights_v)
    collision_u = _task_collision(transfer_u, weights_u)
    mu_v = _terminal_marginal(transfer_v, weights_v)
    mu_u = _terminal_marginal(transfer_u, weights_u)
    mu_v_reference = (transfer_cin**2) @ mu_u

    rows = [
        _row("collision_weight", collision_v, collision_u),
        _row(
            "p2_entropy",
            _entropy_from_collision(collision_v),
            _entropy_from_collision(collision_u),
        ),
        _row("terminal_marginal_linf", float(np.max(np.abs(mu_v - mu_v_reference))), 0.0),
        _row("smooth_support", float(_smooth_support(mu_v, eta)), float(_smooth_support(mu_u, eta))),
        _row(
            "b_col",
            _termwise_support(transfer_v, weights_v, eta),
            _termwise_support(transfer_u, weights_u, eta),
        ),
    ]
    return rows


def _pauli_labels(n_qubits: int) -> list[tuple[int, ...]]:
    return list(product(range(4), repeat=n_qubits))


def _task_weights(labels: list[tuple[int, ...]], smoke: bool) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=float)
    predicates = (
        [
            lambda codes: codes[0] in {1, 3} and codes[1] == 0,
            lambda codes: codes[0] == 0 and codes[1] in {1, 2},
        ]
        if smoke
        else [
            lambda codes: codes[0] in {1, 2} and codes[1] == 0 and codes[2] == 0,
            lambda codes: codes[0] == 0 and codes[1] == 3 and codes[2] in {0, 1},
            lambda codes: codes[0] in {0, 3} and codes[1] in {1, 2} and codes[2] == 0,
        ]
    )
    selected: list[int] = []
    for predicate in predicates:
        selected.extend(index for index, codes in enumerate(labels) if predicate(codes))
    selected = sorted(set(selected))
    if not selected:
        raise ValueError("task selected no Pauli columns")
    for index in selected:
        weights[index] = 1.0 / len(selected)
    return weights


def _task_collision(transfer: np.ndarray, weights: np.ndarray) -> float:
    column_collisions = np.sum(transfer**4, axis=0)
    return float(np.dot(weights, column_collisions))


def _terminal_marginal(transfer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum((transfer**2) * weights[np.newaxis, :], axis=1)


def _entropy_from_collision(collision: float) -> float:
    entropy = -math.log(max(collision, 1e-300))
    return 0.0 if abs(entropy) < 1e-12 else entropy


def _smooth_support(weights: np.ndarray, eta: float) -> int:
    ordered = np.sort(weights)[::-1]
    cumulative = np.cumsum(ordered)
    return int(np.searchsorted(cumulative, eta, side="left") + 1)


def _termwise_support(transfer: np.ndarray, weights: np.ndarray, eta: float) -> float:
    total = 0.0
    for index, weight in enumerate(weights):
        if weight > 0.0:
            total += float(weight) * _smooth_support(transfer[:, index] ** 2, eta)
    return total


def _row(check: str, value: float, reference: float) -> dict[str, object]:
    return {
        "check": check,
        "value": value,
        "reference": reference,
        "abs_error": abs(value - reference),
    }


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
