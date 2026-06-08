from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random

from r41_path_entropy import (
    Gate,
    brickwork_clifford_family,
    gate,
    linear_fit,
    operator_magic_from_gates,
    path_summary,
    simple_clifford_path_family,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41041


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _constructed_rows()
    rows.extend(_sample_rows())

    scatter_path = RESULTS_DIR / "r41_scatter_path_vs_magic.csv"
    _write_csv(scatter_path, rows)

    extremes = _find_extremes(rows)
    extremes_path = RESULTS_DIR / "r41_extreme_examples.json"
    extremes_path.write_text(json.dumps(extremes, indent=2), encoding="utf-8")

    growth = _growth_fits()
    growth_path = RESULTS_DIR / "r41_growth_fits.json"
    growth_path.write_text(json.dumps(growth, indent=2), encoding="utf-8")

    report = {
        "num_rows": len(rows),
        "scatter_csv": str(scatter_path),
        "extreme_examples_json": str(extremes_path),
        "growth_fits_json": str(growth_path),
        "extremes": extremes,
        "growth": growth,
    }
    print(json.dumps(report, indent=2))


def _constructed_rows() -> list[dict[str, object]]:
    cases: list[tuple[str, int, int, list[Gate]]] = []
    for n_qubits in [1, 2, 3]:
        cases.append((f"T_layer_n{n_qubits}", n_qubits, 1, [gate("T", q) for q in range(n_qubits)]))
        cases.append((f"H_layer_n{n_qubits}", n_qubits, 1, [gate("H", q) for q in range(n_qubits)]))
        for depth in [1, 2, 3]:
            cases.append(
                (
                    f"simple_clifford_n{n_qubits}_L{depth}",
                    n_qubits,
                    depth,
                    simple_clifford_path_family(n_qubits, depth),
                )
            )
    for depth in [1, 2, 3, 4]:
        cases.append((f"brickwork_clifford_n2_L{depth}", 2, depth, brickwork_clifford_family(2, depth)))

    return [_measure_case(label, n_qubits, depth, gates, "constructed") for label, n_qubits, depth, gates in cases]


def _sample_rows() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED)
    rows: list[dict[str, object]] = []
    specs = {
        1: {"max_depth": 12, "samples": 30},
        2: {"max_depth": 8, "samples": 40},
        3: {"max_depth": 6, "samples": 12},
    }
    for n_qubits, spec in specs.items():
        available = _gate_set(n_qubits)
        for depth in range(1, int(spec["max_depth"]) + 1):
            seen: set[str] = set()
            attempts = 0
            while len(seen) < int(spec["samples"]) and attempts < int(spec["samples"]) * 20:
                attempts += 1
                gates = [rng.choice(available) for _ in range(depth)]
                key = _format_circuit(gates)
                if key in seen:
                    continue
                seen.add(key)
                label = f"sample_n{n_qubits}_d{depth}_{len(seen):03d}"
                rows.append(_measure_case(label, n_qubits, depth, gates, "sample"))
    return rows


def _measure_case(
    label: str,
    n_qubits: int,
    depth: int,
    gates: list[Gate],
    source: str,
) -> dict[str, object]:
    summary = path_summary(n_qubits, gates, max_total_paths=5_000_000)
    magic = operator_magic_from_gates(n_qubits, gates)
    return {
        "label": label,
        "source": source,
        "n_qubits": n_qubits,
        "depth": depth,
        "num_gates": len(gates),
        "circuit": _format_circuit(gates),
        "median_c2": summary.median_c2,
        "mean_c2": summary.mean_c2,
        "max_c2": summary.max_c2,
        "median_cancellation": summary.median_cancellation,
        "operator_magic": magic,
    }


def _gate_set(n_qubits: int) -> list[Gate]:
    gates: list[Gate] = []
    for q in range(n_qubits):
        gates.extend([gate("H", q), gate("S", q), gate("T", q)])
    for q in range(n_qubits - 1):
        gates.extend([gate("CNOT", q, q + 1), gate("CNOT", q + 1, q), gate("CZ", q, q + 1)])
    return gates


def _find_extremes(rows: list[dict[str, object]]) -> dict[str, object]:
    magic_zero = [row for row in rows if float(row["operator_magic"]) < 1e-10]
    path_zero = [row for row in rows if float(row["median_c2"]) < 1e-12]
    return {
        "high_path_low_magic": _public_row(max(magic_zero, key=lambda row: float(row["median_c2"]))),
        "low_path_high_magic": _public_row(max(path_zero, key=lambda row: float(row["operator_magic"]))),
        "max_path": _public_row(max(rows, key=lambda row: float(row["median_c2"]))),
        "max_magic": _public_row(max(rows, key=lambda row: float(row["operator_magic"]))),
    }


def _growth_fits() -> dict[str, object]:
    output: dict[str, object] = {}
    for family_name, builder in [
        ("simple_clifford", simple_clifford_path_family),
        ("brickwork_clifford_n2", brickwork_clifford_family),
    ]:
        depths = [1, 2, 3, 4]
        c2_values = [
            path_summary(2, builder(2, depth), max_total_paths=1_000_000).median_c2
            for depth in depths
        ]
        output[family_name] = {
            "depths": depths,
            "median_c2": c2_values,
            "median_log2_paths": [value / math.log(2.0) for value in c2_values],
            "fit": linear_fit(depths, c2_values),
            "operator_magic_depth_4": operator_magic_from_gates(2, builder(2, 4)),
        }
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "source",
        "n_qubits",
        "depth",
        "num_gates",
        "circuit",
        "median_c2",
        "mean_c2",
        "max_c2",
        "median_cancellation",
        "operator_magic",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _public_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "label": row["label"],
        "source": row["source"],
        "n_qubits": row["n_qubits"],
        "depth": row["depth"],
        "circuit": row["circuit"],
        "median_c2": row["median_c2"],
        "median_cancellation": row["median_cancellation"],
        "operator_magic": row["operator_magic"],
    }


def _format_circuit(gates: list[Gate]) -> str:
    return " ".join(f"{item.name}{','.join(str(q) for q in item.qubits)}" for item in gates)


if __name__ == "__main__":
    main()
