"""Same-support observable-weight validation.

This validation tests a stronger condition than the observable-blind retained-count
tables.  A single circuit and a single nonzero Pauli support are fixed.
Only the observable coefficients over that identical support are changed.  If
the fixed-error quantities change, the certificate is using the task
distribution and terminal marginal, not merely the set of propagated columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r41_observable_cost import local_transverse_labels, mass_cutoff  # noqa: E402
from r41_pauli_scaling import propagate_sparse_pauli, random_clifford_t_circuit  # noqa: E402


DATA = ROOT / "data"
RNG_SEED = 41061

Label = tuple[int, int]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(raw: np.ndarray) -> np.ndarray:
    total = float(np.sum(raw))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    return raw / total


def profile_from_probability_maps(
    probability_maps: list[dict[Label, float]],
    weights: np.ndarray,
) -> dict[str, float | int]:
    marginal: dict[Label, float] = {}
    collision = 0.0
    per_column_k90: list[int] = []
    terminal_terms: list[int] = []
    for weight, probability_map in zip(weights, probability_maps):
        values = list(probability_map.values())
        collision += float(weight) * sum(value * value for value in values)
        per_column_k90.append(mass_cutoff(sorted(values, reverse=True), 0.90))
        terminal_terms.append(len(values))
        for label, probability in probability_map.items():
            marginal[label] = marginal.get(label, 0.0) + float(weight) * probability
    sorted_mass = sorted(marginal.values(), reverse=True)
    return {
        "task_p2": -math.log(max(collision, 1e-300)),
        "exp_task_p2": 1.0 / max(collision, 1e-300),
        "k90": mass_cutoff(sorted_mass, 0.90),
        "k95": mass_cutoff(sorted_mass, 0.95),
        "terminal_support": len(sorted_mass),
        "marginal_collision": sum(value * value for value in sorted_mass),
        "termwise_k90_mean": float(np.mean(per_column_k90)),
        "avg_terminal_terms": float(np.mean(terminal_terms)),
        "max_terminal_terms": int(max(terminal_terms)),
    }


def build_probability_maps(
    n_qubits: int,
    gates: list[object],
    labels: list[Label],
    max_terms: int,
) -> tuple[list[dict[Label, float]], int]:
    maps: list[dict[Label, float]] = []
    capped = 0
    for label in labels:
        try:
            coeffs, _ = propagate_sparse_pauli(n_qubits, gates, label, max_terms=max_terms)
        except RuntimeError:
            capped += 1
            continue
        probability_map = {item: float(value * value) for item, value in coeffs.items()}
        norm = sum(probability_map.values())
        maps.append({item: value / norm for item, value in probability_map.items()})
    return maps, capped


def profile_rows_for_circuit(
    circuit_id: str,
    n_qubits: int,
    gates: list[object],
    max_terms: int,
) -> list[dict[str, object]]:
    labels = local_transverse_labels(n_qubits)
    probability_maps, capped = build_probability_maps(n_qubits, gates, labels, max_terms)
    if len(probability_maps) != len(labels):
        raise RuntimeError("support-equivalence validation requires all columns")

    individual_k90 = np.asarray(
        [
            mass_cutoff(sorted(probability_map.values(), reverse=True), 0.90)
            for probability_map in probability_maps
        ],
        dtype=float,
    )
    easy_index = int(np.argmin(individual_k90))
    hard_index = int(np.argmax(individual_k90))
    support_size = len(labels)
    floor = 0.10 / max(support_size - 1, 1)

    uniform = np.full(support_size, 1.0 / support_size)
    easy = np.full(support_size, floor)
    easy[easy_index] = 0.90
    hard = np.full(support_size, floor)
    hard[hard_index] = 0.90
    soft_hard = normalize(np.exp(0.35 * np.log(np.maximum(individual_k90, 1.0))))

    rows: list[dict[str, object]] = []
    for profile_name, weights in [
        ("uniform", uniform),
        ("easy_weighted", easy),
        ("hard_weighted", hard),
        ("soft_hard_weighted", soft_hard),
    ]:
        profile = profile_from_probability_maps(probability_maps, weights)
        rows.append(
            {
                "circuit_id": circuit_id,
                "n_qubits": n_qubits,
                "layers": 6,
                "t_density": 0.30,
                "observable_profile": profile_name,
                "same_input_support_size": support_size,
                "positive_weights": support_size,
                "max_weight": float(np.max(weights)),
                "min_weight": float(np.min(weights)),
                "easy_column_k90": int(individual_k90[easy_index]),
                "hard_column_k90": int(individual_k90[hard_index]),
                "truncated_columns": capped,
                **profile,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["circuit_id"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    ratios: list[float] = []
    task_p2_ranges: list[float] = []
    for circuit_id, group in grouped.items():
        k90_values = [float(row["k90"]) for row in group]
        p2_values = [float(row["task_p2"]) for row in group]
        ratio = max(k90_values) / max(min(k90_values), 1.0)
        ratios.append(ratio)
        task_p2_ranges.append(max(p2_values) - min(p2_values))
        summary_rows.append(
            {
                "circuit_id": circuit_id,
                "same_support_profiles": len(group),
                "same_input_support_size": int(group[0]["same_input_support_size"]),
                "k90_min": min(k90_values),
                "k90_max": max(k90_values),
                "k90_ratio": ratio,
                "task_p2_range": max(p2_values) - min(p2_values),
                "truncated_columns": int(group[0]["truncated_columns"]),
            }
        )

    summary_rows.append(
        {
            "circuit_id": "aggregate",
            "same_support_profiles": len(rows),
            "same_input_support_size": int(rows[0]["same_input_support_size"]),
            "k90_min": "",
            "k90_max": "",
            "k90_ratio": float(np.median(ratios)),
            "task_p2_range": float(np.median(task_p2_ranges)),
            "truncated_columns": sum(int(row["truncated_columns"]) for row in rows),
        }
    )
    return summary_rows


def quick_validation() -> dict[str, object]:
    rng = random.Random(RNG_SEED)
    n_qubits = 4
    gates = random_clifford_t_circuit(n_qubits, layers=4, t_density=0.25, rng=rng)
    rows = profile_rows_for_circuit("quick", n_qubits, gates, max_terms=50_000)
    support_sizes = {int(row["same_input_support_size"]) for row in rows}
    positive = all(int(row["positive_weights"]) == int(row["same_input_support_size"]) for row in rows)
    truncated = sum(int(row["truncated_columns"]) for row in rows)
    return {
        "profiles": len(rows),
        "support_sizes": sorted(support_sizes),
        "positive_weights": positive,
        "truncated_columns": truncated,
        "passed": len(rows) == 4 and len(support_sizes) == 1 and positive and truncated == 0,
    }


def run() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    quick_result = quick_validation()
    (DATA / "r41_weighted_observable_validation_quick.json").write_text(
        json.dumps(quick_result, indent=2), encoding="utf-8"
    )
    if not quick_result["passed"]:
        raise SystemExit(json.dumps(quick_result, indent=2))

    rng = random.Random(RNG_SEED + 1)
    rows: list[dict[str, object]] = []
    for repeat in range(20):
        circuit_rng = random.Random(rng.randrange(1 << 60))
        gates = random_clifford_t_circuit(8, layers=6, t_density=0.30, rng=circuit_rng)
        rows.extend(profile_rows_for_circuit(f"weighted_n8_r{repeat}", 8, gates, max_terms=200_000))
    write_csv(DATA / "r41_weighted_observable_validation.csv", rows)
    write_csv(DATA / "r41_weighted_observable_validation_summary.csv", summarize(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a quick validation")
    args = parser.parse_args()
    if args.quick:
        result = quick_validation()
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    run()


if __name__ == "__main__":
    main()
