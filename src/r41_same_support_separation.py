"""Analytic same-support coefficient separation.

The experiment fixes one channel and one nonzero Pauli support.  Two
observables differ only by positive coefficient weights on that same support.
The resulting fixed-error retained support and terminal-collision scales are
computed from closed-form T^{\otimes n} formulas.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def separation_row(n_qubits: int, eta: Fraction) -> dict[str, object]:
    paths = 2**n_qubits
    gamma = Fraction(1, paths)
    transverse_collision = gamma

    p_easy = (1 - gamma) + gamma * transverse_collision
    p_hard = gamma + (1 - gamma) * transverse_collision

    k_easy = 1
    k_hard = ceil_fraction(((eta - gamma) / (1 - gamma)) * paths)

    return {
        "n_qubits": n_qubits,
        "eta": float(eta),
        "gamma": float(gamma),
        "same_nonzero_pauli_support": 2,
        "positive_weights": True,
        "operator_sre": n_qubits * math.log(4.0 / 3.0),
        "k_eta_easy_weighted": k_easy,
        "k_eta_hard_weighted": k_hard,
        "k_eta_ratio": float(k_hard / k_easy),
        "p_easy_weighted": float(p_easy),
        "p_hard_weighted": float(p_hard),
        "collision_scale_ratio": float(p_easy / p_hard),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quick_validation() -> dict[str, object]:
    eta = Fraction(9, 10)
    row = separation_row(4, eta)
    passed = (
        row["same_nonzero_pauli_support"] == 2
        and row["positive_weights"] is True
        and row["k_eta_hard_weighted"] > row["k_eta_easy_weighted"]
        and row["collision_scale_ratio"] > 1.0
    )
    return {
        "n_qubits": row["n_qubits"],
        "eta": row["eta"],
        "k_eta_ratio": row["k_eta_ratio"],
        "collision_scale_ratio": row["collision_scale_ratio"],
        "passed": passed,
    }


def run(n_values: list[int], eta: Fraction) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    quick_result = quick_validation()
    (DATA / "r41_same_support_separation_quick.json").write_text(
        json.dumps(quick_result, indent=2), encoding="utf-8"
    )
    if not quick_result["passed"]:
        raise SystemExit(json.dumps(quick_result, indent=2))
    rows = [separation_row(n_qubits, eta) for n_qubits in n_values]
    write_csv(DATA / "r41_same_support_separation.csv", rows)


def parse_eta(text: str) -> Fraction:
    return Fraction(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a quick validation")
    parser.add_argument("--eta", type=parse_eta, default=Fraction(9, 10))
    parser.add_argument(
        "--n-values",
        nargs="*",
        type=int,
        default=[8, 16, 32, 64, 96],
    )
    args = parser.parse_args()
    if args.quick:
        result = quick_validation()
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    run(args.n_values, args.eta)


if __name__ == "__main__":
    main()
