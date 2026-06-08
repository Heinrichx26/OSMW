from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

from r41_path_entropy import gate, pauli_transfer_matrix, unitary_from_gates


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate task mixture and tensor-product calculus laws."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a minimal validation before the full grid",
    )
    args = parser.parse_args()
    quick = args.smoke or args.quick

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = _smoke_rows() if quick else _full_rows()
    validation = _validation_payload(rows, smoke=quick)
    if quick:
        _write_json(DATA_DIR / "r41_task_calculus_quick.json", validation)
    else:
        _write_csv(DATA_DIR / "r41_task_calculus_checks.csv", rows)
        _write_json(DATA_DIR / "r41_task_calculus_checks.json", {"rows": rows})
        _write_json(DATA_DIR / "r41_task_calculus_validation.json", validation)
    print(json.dumps(validation, indent=2))


def _smoke_rows() -> list[dict[str, object]]:
    transfer = pauli_transfer_matrix(
        unitary_from_gates(2, [gate("T", 0), gate("H", 1), gate("T", 1)])
    )
    labels = _pauli_labels(2)
    weights_a = _uniform_task(labels, lambda codes: codes[0] == 3)
    weights_b = _uniform_task(labels, lambda codes: codes[0] in {1, 2})
    return _mixture_rows(transfer, weights_a, weights_b, lam=0.4, eta_a=0.9, eta_b=0.8)


def _full_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    transfer = pauli_transfer_matrix(
        unitary_from_gates(
            3,
            [
                gate("H", 0),
                gate("T", 0),
                gate("CNOT", 0, 1),
                gate("S", 1),
                gate("T", 2),
                gate("CZ", 1, 2),
                gate("H", 2),
                gate("T", 1),
            ],
        )
    )
    labels = _pauli_labels(3)
    weights_a = _uniform_task(
        labels,
        lambda codes: codes[0] == 3 and all(code == 0 for code in codes[1:]),
    )
    weights_b = _uniform_task(labels, lambda codes: codes[0] in {1, 2})
    rows.extend(_mixture_rows(transfer, weights_a, weights_b, lam=0.37, eta_a=0.9, eta_b=0.82))
    rows.extend(_product_rows())
    return rows


def _mixture_rows(
    transfer: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    lam: float,
    eta_a: float,
    eta_b: float,
) -> list[dict[str, object]]:
    weights_mix = lam * weights_a + (1.0 - lam) * weights_b
    p_a = _task_collision(transfer, weights_a)
    p_b = _task_collision(transfer, weights_b)
    p_mix = _task_collision(transfer, weights_mix)
    mu_a = _terminal_marginal(transfer, weights_a)
    mu_b = _terminal_marginal(transfer, weights_b)
    mu_mix = _terminal_marginal(transfer, weights_mix)
    mu_ref = lam * mu_a + (1.0 - lam) * mu_b
    eta_mix = lam * eta_a + (1.0 - lam) * eta_b
    support_bound = _smooth_support(mu_a, eta_a) + _smooth_support(mu_b, eta_b)
    support_mix = _smooth_support(mu_mix, eta_mix)
    entropy_mix = _entropy_from_collision(p_mix)
    entropy_ref = -math.log(lam * math.exp(-_entropy_from_collision(p_a)) + (1.0 - lam) * math.exp(-_entropy_from_collision(p_b)))
    return [
        _equality_row("mixture_collision_affine", p_mix, lam * p_a + (1.0 - lam) * p_b),
        _equality_row("mixture_entropy_logsum", entropy_mix, entropy_ref),
        _equality_row("mixture_terminal_marginal_linf", float(np.max(np.abs(mu_mix - mu_ref))), 0.0),
        _bound_row("mixture_smooth_support_union", float(support_mix), float(support_bound)),
    ]


def _product_rows() -> list[dict[str, object]]:
    transfer_a = pauli_transfer_matrix(unitary_from_gates(1, [gate("H", 0), gate("T", 0)]))
    transfer_b = pauli_transfer_matrix(
        unitary_from_gates(2, [gate("T", 0), gate("CNOT", 0, 1), gate("S", 1)])
    )
    transfer_total = pauli_transfer_matrix(
        unitary_from_gates(
            3,
            [
                gate("H", 0),
                gate("T", 0),
                gate("T", 1),
                gate("CNOT", 1, 2),
                gate("S", 2),
            ],
        )
    )
    labels_a = _pauli_labels(1)
    labels_b = _pauli_labels(2)
    labels_total = _pauli_labels(3)
    weights_a = _uniform_task(labels_a, lambda codes: codes[0] in {1, 2})
    weights_b = _uniform_task(labels_b, lambda codes: codes[0] in {0, 3})
    weights_total = _product_task(labels_total, labels_a, labels_b, weights_a, weights_b)
    p_a = _task_collision(transfer_a, weights_a)
    p_b = _task_collision(transfer_b, weights_b)
    p_total = _task_collision(transfer_total, weights_total)
    mu_a = _terminal_marginal(transfer_a, weights_a)
    mu_b = _terminal_marginal(transfer_b, weights_b)
    mu_total = _terminal_marginal(transfer_total, weights_total)
    mu_ref = _product_marginal(labels_total, labels_a, labels_b, mu_a, mu_b)
    eta_a = 0.9
    eta_b = 0.85
    support_total = _smooth_support(mu_total, eta_a * eta_b)
    support_bound = _smooth_support(mu_a, eta_a) * _smooth_support(mu_b, eta_b)
    return [
        _equality_row("product_collision_multiplicative", p_total, p_a * p_b),
        _equality_row(
            "product_entropy_additive",
            _entropy_from_collision(p_total),
            _entropy_from_collision(p_a) + _entropy_from_collision(p_b),
        ),
        _equality_row("product_terminal_marginal_linf", float(np.max(np.abs(mu_total - mu_ref))), 0.0),
        _bound_row("product_smooth_support_bound", float(support_total), float(support_bound)),
    ]


def _pauli_labels(n_qubits: int) -> list[tuple[int, ...]]:
    return list(product(range(4), repeat=n_qubits))


def _uniform_task(
    labels: list[tuple[int, ...]],
    predicate: Callable[[tuple[int, ...]], bool],
) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=float)
    selected = [index for index, codes in enumerate(labels) if predicate(codes)]
    if not selected:
        raise ValueError("task predicate selected no Pauli labels")
    for index in selected:
        weights[index] = 1.0 / len(selected)
    return weights


def _product_task(
    labels_total: list[tuple[int, ...]],
    labels_a: list[tuple[int, ...]],
    labels_b: list[tuple[int, ...]],
    weights_a: np.ndarray,
    weights_b: np.ndarray,
) -> np.ndarray:
    index_a = {codes: index for index, codes in enumerate(labels_a)}
    index_b = {codes: index for index, codes in enumerate(labels_b)}
    weights = np.zeros(len(labels_total), dtype=float)
    for index, codes in enumerate(labels_total):
        weights[index] = weights_a[index_a[codes[:1]]] * weights_b[index_b[codes[1:]]]
    return weights


def _product_marginal(
    labels_total: list[tuple[int, ...]],
    labels_a: list[tuple[int, ...]],
    labels_b: list[tuple[int, ...]],
    mu_a: np.ndarray,
    mu_b: np.ndarray,
) -> np.ndarray:
    index_a = {codes: index for index, codes in enumerate(labels_a)}
    index_b = {codes: index for index, codes in enumerate(labels_b)}
    mu = np.zeros(len(labels_total), dtype=float)
    for index, codes in enumerate(labels_total):
        mu[index] = mu_a[index_a[codes[:1]]] * mu_b[index_b[codes[1:]]]
    return mu


def _task_collision(transfer: np.ndarray, weights: np.ndarray) -> float:
    column_collisions = np.sum(transfer**4, axis=0)
    return float(np.dot(weights, column_collisions))


def _terminal_marginal(transfer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum((transfer**2) * weights[np.newaxis, :], axis=1)


def _entropy_from_collision(collision: float) -> float:
    entropy = -math.log(max(collision, 1e-300))
    return 0.0 if abs(entropy) < 1e-12 else entropy


def _smooth_support(mu: np.ndarray, eta: float) -> int:
    if eta <= 0.0 or eta > 1.0:
        raise ValueError("eta must lie in (0, 1]")
    ordered = np.sort(mu)[::-1]
    cumulative = np.cumsum(ordered)
    return int(np.searchsorted(cumulative, eta, side="left") + 1)


def _equality_row(check: str, value: float, reference: float) -> dict[str, object]:
    return {
        "check": check,
        "value": value,
        "reference": reference,
        "abs_error": abs(value - reference),
        "slack": "",
    }


def _bound_row(check: str, value: float, reference: float) -> dict[str, object]:
    return {
        "check": check,
        "value": value,
        "reference": reference,
        "abs_error": "",
        "slack": reference - value,
    }


def _validation_payload(rows: list[dict[str, object]], smoke: bool) -> dict[str, object]:
    errors = [float(row["abs_error"]) for row in rows if row["abs_error"] != ""]
    slacks = [float(row["slack"]) for row in rows if row["slack"] != ""]
    max_abs_error = max(errors) if errors else 0.0
    min_slack = min(slacks) if slacks else 0.0
    return {
        "status": "passed" if max_abs_error < 1e-12 and min_slack >= -1e-12 else "failed",
        "quick": smoke,
        "num_rows": len(rows),
        "max_abs_error": max_abs_error,
        "min_slack": min_slack,
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
