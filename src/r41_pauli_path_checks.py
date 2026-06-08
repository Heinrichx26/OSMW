from __future__ import annotations

import json
import math
from pathlib import Path
import random

from r41_path_entropy import (
    Gate,
    brickwork_clifford_family,
    gate,
    operator_magic_from_gates,
    pauli_path_coherence_from_gates,
    simple_clifford_path_family,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41042


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rows.extend(_constructed_rows())
    rows.extend(_random_rows())

    output = {
        "definition": "P2(U) = -log(4^{-n} sum_{P,Q} R_QP(U)^4)",
        "rows": rows,
        "max_abs_difference": max(abs(float(row["difference"])) for row in rows),
    }
    output_path = RESULTS_DIR / "r41_pauli_path_checks.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _constructed_rows() -> list[dict[str, object]]:
    cases: list[tuple[str, int, list[Gate]]] = [
        ("H", 1, [gate("H", 0)]),
        ("T", 1, [gate("T", 0)]),
        ("T_tensor_2", 2, [gate("T", 0), gate("T", 1)]),
        ("T_tensor_3", 3, [gate("T", 0), gate("T", 1), gate("T", 2)]),
        ("simple_clifford_n2_L3", 2, simple_clifford_path_family(2, 3)),
        ("brickwork_clifford_n2_L3", 2, brickwork_clifford_family(2, 3)),
        ("HTCNOT", 2, [gate("H", 0), gate("T", 0), gate("CNOT", 0, 1)]),
    ]
    return [_measure(label, n_qubits, gates, "constructed") for label, n_qubits, gates in cases]


def _random_rows() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED)
    rows: list[dict[str, object]] = []
    for n_qubits, samples, depth in [(1, 12, 8), (2, 12, 8), (3, 6, 6)]:
        gates = _gate_set(n_qubits)
        for index in range(samples):
            circuit = [rng.choice(gates) for _ in range(depth)]
            rows.append(_measure(f"random_n{n_qubits}_{index:02d}", n_qubits, circuit, "random"))
    return rows


def _measure(label: str, n_qubits: int, gates: list[Gate], source: str) -> dict[str, object]:
    p2 = pauli_path_coherence_from_gates(n_qubits, gates)
    magic = operator_magic_from_gates(n_qubits, gates)
    return {
        "label": label,
        "source": source,
        "n_qubits": n_qubits,
        "num_gates": len(gates),
        "circuit": _format_circuit(gates),
        "pauli_path_coherence": p2,
        "operator_magic": magic,
        "difference": p2 - magic,
        "expected_t_multiple": p2 / math.log(4.0 / 3.0) if p2 > 1e-12 else 0.0,
    }


def _gate_set(n_qubits: int) -> list[Gate]:
    gates: list[Gate] = []
    for q in range(n_qubits):
        gates.extend([gate("H", q), gate("S", q), gate("T", q)])
    for q in range(n_qubits - 1):
        gates.extend([gate("CNOT", q, q + 1), gate("CNOT", q + 1, q), gate("CZ", q, q + 1)])
    return gates


def _format_circuit(gates: list[Gate]) -> str:
    return " ".join(f"{item.name}{','.join(str(q) for q in item.qubits)}" for item in gates)


if __name__ == "__main__":
    main()
