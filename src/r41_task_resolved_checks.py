from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random

import numpy as np

from r41_path_entropy import (
    Gate,
    gate,
    operator_magic_from_gates,
    pauli_path_coherence_from_gates,
    pauli_transfer_matrix,
    unitary_from_gates,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41044


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n_qubits in [1, 2, 3]:
        rows.extend(_t_tensor_rows(n_qubits))
    rows.extend(_random_task_rows())
    _write_csv(RESULTS_DIR / "r41_task_resolved.csv", rows)
    output = {
        "rows": rows,
        "notes": {
            "z_sector": "Pauli strings with only I/Z factors",
            "xy_all_sector": "Pauli strings with only X/Y factors on every site",
            "xy_one_site_sector": "Pauli strings with X/Y on one chosen site and I elsewhere",
        },
    }
    output_path = RESULTS_DIR / "r41_task_resolved.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _t_tensor_rows(n_qubits: int) -> list[dict[str, object]]:
    gates = [gate("T", q) for q in range(n_qubits)]
    transfer = pauli_transfer_matrix(unitary_from_gates(n_qubits, gates))
    sectors = {
        "all_paulis": _all_indices(n_qubits),
        "z_sector": _z_sector(n_qubits),
        "xy_all_sector": _xy_all_sector(n_qubits),
        "xy_one_site_sector": _xy_one_site_sector(n_qubits, site=0),
    }
    rows = []
    for sector, indices in sectors.items():
        p2 = task_entropy_from_transfer(transfer, indices)
        rows.append(
            {
                "family": "T_tensor",
                "n_qubits": n_qubits,
                "task": sector,
                "omega_size": len(indices),
                "task_p2": p2,
                "operator_magic": operator_magic_from_gates(n_qubits, gates),
                "global_pauli_p2": pauli_path_coherence_from_gates(n_qubits, gates),
                "analytic": _analytic_t_value(n_qubits, sector),
            }
        )
    return rows


def _random_task_rows() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED)
    rows: list[dict[str, object]] = []
    for index in range(4):
        n_qubits = 3
        gates = _random_circuit(n_qubits, depth=8, rng=rng)
        transfer = pauli_transfer_matrix(unitary_from_gates(n_qubits, gates))
        for sector, indices in {
            "all_paulis": _all_indices(n_qubits),
            "local_observables": _single_site_nonidentity(n_qubits),
            "z_sector": _z_sector(n_qubits),
        }.items():
            rows.append(
                {
                    "family": "random_clifford_t",
                    "n_qubits": n_qubits,
                    "task": sector,
                    "omega_size": len(indices),
                    "task_p2": task_entropy_from_transfer(transfer, indices),
                    "operator_magic": operator_magic_from_gates(n_qubits, gates),
                    "global_pauli_p2": pauli_path_coherence_from_gates(n_qubits, gates),
                    "analytic": "",
                }
            )
    return rows


def task_entropy_from_transfer(transfer: np.ndarray, omega: list[int]) -> float:
    if not omega:
        raise ValueError("omega must be nonempty")
    value = float(np.mean([np.sum(transfer[:, p_index] ** 4) for p_index in omega]))
    entropy = -math.log(max(value, 1e-300))
    return 0.0 if abs(entropy) < 1e-12 else entropy


def _all_indices(n_qubits: int) -> list[int]:
    return list(range(4**n_qubits))


def _z_sector(n_qubits: int) -> list[int]:
    return [_pauli_index(codes) for codes in _product_codes([0, 3], n_qubits)]


def _xy_all_sector(n_qubits: int) -> list[int]:
    return [_pauli_index(codes) for codes in _product_codes([1, 2], n_qubits)]


def _xy_one_site_sector(n_qubits: int, site: int) -> list[int]:
    rows = []
    for code in [1, 2]:
        codes = [0] * n_qubits
        codes[site] = code
        rows.append(_pauli_index(codes))
    return rows


def _single_site_nonidentity(n_qubits: int) -> list[int]:
    rows = []
    for site in range(n_qubits):
        for code in [1, 2, 3]:
            codes = [0] * n_qubits
            codes[site] = code
            rows.append(_pauli_index(codes))
    return rows


def _product_codes(options: list[int], n_qubits: int) -> list[list[int]]:
    if n_qubits == 0:
        return [[]]
    suffixes = _product_codes(options, n_qubits - 1)
    return [[code] + suffix for code in options for suffix in suffixes]


def _pauli_index(codes: list[int]) -> int:
    index = 0
    for q, code in enumerate(codes):
        index += code * (4**q)
    return index


def _analytic_t_value(n_qubits: int, sector: str) -> str:
    if sector == "all_paulis":
        return f"{n_qubits} log(4/3)"
    if sector == "z_sector":
        return "0"
    if sector == "xy_all_sector":
        return f"{n_qubits} log 2"
    if sector == "xy_one_site_sector":
        return "log 2"
    return ""


def _random_circuit(n_qubits: int, depth: int, rng: random.Random) -> list[Gate]:
    gates: list[Gate] = []
    for layer in range(depth):
        for q in range(n_qubits):
            draw = rng.random()
            if draw < 0.35:
                gates.append(gate("H", q))
            elif draw < 0.70:
                gates.append(gate("S", q))
            else:
                gates.append(gate("T", q))
        q0 = layer % (n_qubits - 1)
        gates.append(gate("CNOT", q0, q0 + 1))
    return gates


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
