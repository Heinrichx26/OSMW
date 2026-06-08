from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from r41_path_entropy import (
    Gate,
    brickwork_clifford_family,
    count_hadamards,
    linear_fit,
    operator_magic_from_gates,
    path_summary,
    simple_clifford_path_family,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MAX_TOTAL_PATHS = 5_000_000


FamilyBuilder = Callable[[int, int], list[Gate]]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    families: dict[str, FamilyBuilder] = {
        "simple_clifford": simple_clifford_path_family,
        "brickwork_clifford": brickwork_clifford_family,
    }

    output = {
        name: _check_family(name, builder)
        for name, builder in families.items()
    }

    output_path = RESULTS_DIR / "r41_family_checks.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _check_family(name: str, builder: FamilyBuilder) -> dict[str, object]:
    by_n: dict[str, object] = {}
    for n_qubits in [1, 2, 3, 4]:
        rows = []
        for depth in range(1, 7):
            gates = builder(n_qubits, depth)
            h_count = count_hadamards(gates)
            total_paths = (1 << n_qubits) * (1 << h_count)
            if total_paths > MAX_TOTAL_PATHS:
                break
            summary = path_summary(
                n_qubits,
                gates,
                max_total_paths=MAX_TOTAL_PATHS,
            )
            rows.append(
                {
                    "depth": depth,
                    "hadamards": h_count,
                    "paths_per_input": summary.paths_per_input,
                    "median_c2": summary.median_c2,
                    "mean_c2": summary.mean_c2,
                    "max_c2": summary.max_c2,
                    "median_log2_paths": summary.median_c2 / math.log(2.0),
                    "num_boundaries": summary.num_boundaries,
                }
            )
        fit = None
        if len(rows) >= 2:
            fit = linear_fit(
                [row["depth"] for row in rows],
                [row["median_c2"] for row in rows],
            )
        magic_probe = None
        if rows and n_qubits <= 3:
            probe_depth = rows[-1]["depth"]
            magic_probe = operator_magic_from_gates(n_qubits, builder(n_qubits, probe_depth))
        by_n[str(n_qubits)] = {
            "rows": rows,
            "median_c2_fit": fit,
            "operator_magic_probe": magic_probe,
        }
    return {"family": name, "by_n": by_n}


if __name__ == "__main__":
    main()
