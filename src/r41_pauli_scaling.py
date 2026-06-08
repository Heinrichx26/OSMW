from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from r41_path_entropy import Gate, gate, pauli_path_coherence_from_gates


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41043


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    smoke = _smoke_test()
    scaling_rows = _scaling_experiment()
    _write_csv(RESULTS_DIR / "r41_pauli_scaling.csv", scaling_rows)
    output = {
        "smoke": smoke,
        "scaling": scaling_rows,
        "interpretation": {
            "x_axis": "expected T gates per circuit = n_qubits * layers * t_density",
            "linear_reference_slope": "log(4/3) per T gate",
            "method": "sampled Pauli columns with exact sparse propagation per sampled column",
        },
    }
    output_path = RESULTS_DIR / "r41_pauli_scaling.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _smoke_test() -> dict[str, object]:
    rng = random.Random(RNG_SEED)
    n_qubits = 3
    gates = random_clifford_t_circuit(n_qubits, layers=5, t_density=0.25, rng=rng)
    exact = pauli_path_coherence_from_gates(n_qubits, gates)
    estimate = estimate_pauli_path_coherence(
        n_qubits,
        gates,
        samples=600,
        rng=random.Random(RNG_SEED + 1),
        max_terms=250_000,
    )
    estimate["exact_p2"] = exact
    estimate["abs_error"] = abs(float(estimate["p2_estimate"]) - exact)
    estimate["passed"] = estimate["abs_error"] <= 3.0 * float(estimate["p2_standard_error"]) + 0.08
    return estimate


def _scaling_experiment() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED + 2)
    rows: list[dict[str, object]] = []
    configs = [
        {"n_qubits": 8, "layers": 10, "samples": 80},
        {"n_qubits": 12, "layers": 8, "samples": 70},
        {"n_qubits": 16, "layers": 6, "samples": 60},
    ]
    densities = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    repeats = 8
    for config in configs:
        n_qubits = int(config["n_qubits"])
        layers = int(config["layers"])
        samples = int(config["samples"])
        for density in densities:
            repeat_estimates = []
            for repeat in range(repeats):
                circuit_rng = random.Random(rng.randrange(1 << 60))
                gates = random_clifford_t_circuit(n_qubits, layers, density, circuit_rng)
                estimator_rng = random.Random(rng.randrange(1 << 60))
                estimate = estimate_pauli_path_coherence(
                    n_qubits,
                    gates,
                    samples=samples,
                    rng=estimator_rng,
                    max_terms=250_000,
                )
                repeat_estimates.append(estimate)
            rows.append(_summarize_repeats(n_qubits, layers, density, repeat_estimates))
    return rows


def random_clifford_t_circuit(
    n_qubits: int,
    layers: int,
    t_density: float,
    rng: random.Random,
) -> list[Gate]:
    gates: list[Gate] = []
    for layer in range(layers):
        for q in range(n_qubits):
            draw = rng.random()
            if draw < 0.45:
                gates.append(gate("H", q))
            elif draw < 0.90:
                gates.append(gate("S", q))
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            gates.append(gate("CNOT", q, q + 1))
        for q in range(1 - start, n_qubits - 1, 2):
            gates.append(gate("CZ", q, q + 1))
        for q in range(n_qubits):
            if rng.random() < t_density:
                gates.append(gate("T", q))
    return gates


def estimate_pauli_path_coherence(
    n_qubits: int,
    gates: Iterable[Gate],
    samples: int,
    rng: random.Random,
    max_terms: int,
) -> dict[str, object]:
    gate_list = list(gates)
    collisions: list[float] = []
    term_counts: list[int] = []
    active_splits: list[int] = []
    capped = 0
    for _ in range(samples):
        xmask = rng.randrange(1 << n_qubits)
        zmask = rng.randrange(1 << n_qubits)
        try:
            coeffs, split_count = propagate_sparse_pauli(
                n_qubits,
                gate_list,
                (xmask, zmask),
                max_terms=max_terms,
            )
        except RuntimeError:
            capped += 1
            continue
        collisions.append(sum(value**4 for value in coeffs.values()))
        term_counts.append(len(coeffs))
        active_splits.append(split_count)

    if not collisions:
        raise RuntimeError("all sampled columns exceeded the sparse propagation cap")
    collision_array = np.asarray(collisions, dtype=float)
    mean_collision = float(np.mean(collision_array))
    sem_collision = float(np.std(collision_array, ddof=1) / math.sqrt(len(collision_array))) if len(collision_array) > 1 else 0.0
    p2 = -math.log(max(mean_collision, 1e-300))
    p2_sem = sem_collision / max(mean_collision, 1e-300)
    return {
        "p2_estimate": p2,
        "p2_standard_error": p2_sem,
        "collision_mean": mean_collision,
        "collision_standard_error": sem_collision,
        "samples_requested": samples,
        "samples_used": len(collisions),
        "capped_samples": capped,
        "avg_terminal_terms": float(np.mean(term_counts)),
        "max_terminal_terms": int(max(term_counts)),
        "avg_active_splits": float(np.mean(active_splits)),
    }


def propagate_sparse_pauli(
    n_qubits: int,
    gates: list[Gate],
    initial: tuple[int, int],
    max_terms: int,
) -> tuple[dict[tuple[int, int], float], int]:
    coeffs: dict[tuple[int, int], float] = {initial: 1.0}
    active_splits = 0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for item in reversed(gates):
        next_coeffs: dict[tuple[int, int], float] = {}
        if item.name in {"T", "TDG"}:
            q = item.qubits[0]
            bit = 1 << q
            for (xmask, zmask), coeff in coeffs.items():
                if xmask & bit:
                    active_splits += 1
                    z0 = zmask & ~bit
                    z1 = zmask | bit
                    if item.name == "T":
                        _add(next_coeffs, (xmask, z0), coeff * inv_sqrt2)
                        sign = -1.0 if (zmask & bit) == 0 else 1.0
                        _add(next_coeffs, (xmask, z1), coeff * sign * inv_sqrt2)
                    else:
                        sign = 1.0 if (zmask & bit) == 0 else -1.0
                        _add(next_coeffs, (xmask, z0), coeff * sign * inv_sqrt2)
                        _add(next_coeffs, (xmask, z1), coeff * inv_sqrt2)
                else:
                    _add(next_coeffs, (xmask, zmask), coeff)
        else:
            for label, coeff in coeffs.items():
                _add(next_coeffs, _apply_clifford_label(n_qubits, label, item), coeff)

        coeffs = {label: value for label, value in next_coeffs.items() if abs(value) > 1e-14}
        if len(coeffs) > max_terms:
            raise RuntimeError("sparse propagation cap exceeded")
    return coeffs, active_splits


def _apply_clifford_label(n_qubits: int, label: tuple[int, int], item: Gate) -> tuple[int, int]:
    xmask, zmask = label
    if item.name == "H":
        bit = 1 << item.qubits[0]
        x_bit = xmask & bit
        z_bit = zmask & bit
        xmask = (xmask & ~bit) | z_bit
        zmask = (zmask & ~bit) | x_bit
        return xmask, zmask
    if item.name in {"S", "SDG"}:
        bit = 1 << item.qubits[0]
        if xmask & bit:
            zmask ^= bit
        return xmask, zmask
    if item.name in {"X", "Z"}:
        return xmask, zmask
    if item.name in {"CNOT", "CX"}:
        control, target = item.qubits
        c_bit = 1 << control
        t_bit = 1 << target
        if xmask & c_bit:
            xmask ^= t_bit
        if zmask & t_bit:
            zmask ^= c_bit
        return xmask, zmask
    if item.name == "CZ":
        q0, q1 = item.qubits
        b0 = 1 << q0
        b1 = 1 << q1
        if xmask & b0:
            zmask ^= b1
        if xmask & b1:
            zmask ^= b0
        return xmask, zmask
    raise ValueError(f"unsupported gate in scaling script: {item.name}")


def _add(coeffs: dict[tuple[int, int], float], label: tuple[int, int], value: float) -> None:
    coeffs[label] = coeffs.get(label, 0.0) + value


def _summarize_repeats(
    n_qubits: int,
    layers: int,
    density: float,
    estimates: list[dict[str, object]],
) -> dict[str, object]:
    p2_values = np.asarray([float(item["p2_estimate"]) for item in estimates], dtype=float)
    requested = sum(int(item["samples_requested"]) for item in estimates)
    used = sum(int(item["samples_used"]) for item in estimates)
    capped = sum(int(item["capped_samples"]) for item in estimates)
    return {
        "n_qubits": n_qubits,
        "layers": layers,
        "t_density": density,
        "expected_t_gates": n_qubits * layers * density,
        "p2_mean": float(np.mean(p2_values)),
        "p2_std_over_circuits": float(np.std(p2_values, ddof=1)) if len(p2_values) > 1 else 0.0,
        "p2_per_expected_t": float(np.mean(p2_values) / max(n_qubits * layers * density, 1e-12)),
        "linear_reference": n_qubits * layers * density * math.log(4.0 / 3.0),
        "samples_requested": requested,
        "samples_used": used,
        "capped_samples": capped,
        "avg_terminal_terms": float(np.mean([float(item["avg_terminal_terms"]) for item in estimates])),
        "max_terminal_terms": int(max(int(item["max_terminal_terms"]) for item in estimates)),
        "avg_active_splits": float(np.mean([float(item["avg_active_splits"]) for item in estimates])),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
