from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np

from r41_path_entropy import Gate, gate, pauli_transfer_matrix, unitary_from_gates
from r41_pauli_scaling import (
    _apply_clifford_label,
    propagate_sparse_pauli,
    random_clifford_t_circuit,
)
from r41_task_resolved_checks import task_entropy_from_transfer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RNG_SEED = 41047

Label = tuple[int, int]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation = run_smoke()
    (DATA_DIR / "r41_observable_cost_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.validate or args.smoke:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    task_rows = run_task_scaling()
    kicked_rows = run_kicked_ising()
    global_task_rows = run_global_task_truncation()
    collapse_rows, truncation_rows, observable_error_rows = run_error_collapse()
    write_csv(DATA_DIR / "r41_observable_task_scaling.csv", task_rows)
    write_csv(DATA_DIR / "r41_kicked_ising.csv", kicked_rows)
    write_csv(DATA_DIR / "r41_global_task_truncation.csv", global_task_rows)
    write_csv(DATA_DIR / "r41_estimator_collapse.csv", collapse_rows)
    write_csv(DATA_DIR / "r41_truncation_profile.csv", truncation_rows)
    write_csv(DATA_DIR / "r41_observable_truncation_error.csv", observable_error_rows)
    (DATA_DIR / "r41_observable_cost.json").write_text(
        json.dumps(
            {
                "validation": validation,
                "kicked_ising": kicked_rows,
                "global_task_truncation": global_task_rows,
                "task_scaling": task_rows,
                "estimator_collapse": collapse_rows,
                "truncation_profile": truncation_rows,
                "observable_truncation_error": observable_error_rows,
                "notes": {
                    "task_p2": "-log mean_P sum_Q R_QP^4",
                    "relative_rmse": "root mean squared relative error of the collision estimator",
                    "observable_truncation_error": "product-state response error after retaining terminal path mass",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_smoke() -> dict[str, object]:
    checks: list[dict[str, object]] = []
    n_qubits = 4
    t_gates = [gate("T", q) for q in range(n_qubits)]
    for task, labels, expected in [
        ("local_z", local_z_labels(n_qubits), 0.0),
        ("local_transverse", local_transverse_labels(n_qubits), math.log(2.0)),
        ("full_transverse", full_transverse_labels(n_qubits), n_qubits * math.log(2.0)),
        ("all_pauli", all_labels(n_qubits), n_qubits * math.log(4.0 / 3.0)),
    ]:
        observed = exact_sparse_task_p2(n_qubits, t_gates, labels, max_terms=50_000)
        checks.append(
            {
                "case": f"T_tensor_{task}",
                "observed": observed,
                "expected": expected,
                "abs_error": abs(observed - expected),
            }
        )

    rng = random.Random(RNG_SEED)
    n_exact = 3
    gates = random_clifford_t_circuit(n_exact, layers=5, t_density=0.25, rng=rng)
    transfer = pauli_transfer_matrix(unitary_from_gates(n_exact, gates))
    for task, labels in [
        ("all_pauli", all_labels(n_exact)),
        ("local_z", local_z_labels(n_exact)),
        ("local_transverse", local_transverse_labels(n_exact)),
    ]:
        exact = task_entropy_from_transfer(transfer, [label_to_index(label) for label in labels])
        sparse = exact_sparse_task_p2(n_exact, gates, labels, max_terms=50_000)
        checks.append(
            {
                "case": f"exact_ptm_vs_sparse_{task}",
                "observed": sparse,
                "expected": exact,
                "abs_error": abs(sparse - exact),
            }
        )

    tdg_gates = [
        gate("H", 0),
        gate("CNOT", 0, 1),
        gate("TDG", 1),
        gate("SDG", 0),
        gate("X", 1),
        gate("T", 0),
    ]
    tdg_transfer = pauli_transfer_matrix(unitary_from_gates(2, tdg_gates))
    tdg_labels = all_labels(2)
    tdg_exact = task_entropy_from_transfer(
        tdg_transfer,
        [label_to_index(label) for label in tdg_labels],
    )
    tdg_sparse = exact_sparse_task_p2(2, tdg_gates, tdg_labels, max_terms=50_000)
    checks.append(
        {
            "case": "exact_ptm_vs_sparse_tdg_sdg_x",
            "observed": tdg_sparse,
            "expected": tdg_exact,
            "abs_error": abs(tdg_sparse - tdg_exact),
        }
    )

    n_dressed = 4
    cout = random_clifford_t_circuit(
        n_dressed, layers=3, t_density=0.0, rng=random.Random(RNG_SEED + 2)
    )
    dressed_gates = [gate("T", q) for q in range(n_dressed)] + cout
    local_labels = local_z_labels(n_dressed)
    transported_labels = [transport_label(n_dressed, label, cout) for label in local_labels]
    dressed = exact_sparse_task_p2(n_dressed, dressed_gates, local_labels, max_terms=50_000)
    transported = exact_sparse_task_p2(
        n_dressed, [gate("T", q) for q in range(n_dressed)], transported_labels, max_terms=50_000
    )
    checks.append(
        {
            "case": "clifford_dressed_separation",
            "observed": dressed,
            "expected": transported,
            "abs_error": abs(dressed - transported),
        }
    )

    terminal_columns = build_terminal_columns(
        n_exact,
        gates,
        local_transverse_labels(n_exact),
        max_terms=50_000,
    )
    collision_check = estimate_task_p2(
        n_exact, gates, local_transverse_labels(n_exact), max_terms=50_000
    )
    checks.append(
        {
            "case": "terminal_distribution_precompute",
            "observed": terminal_columns["collision_probability"],
            "expected": collision_check["collision_mean"],
            "abs_error": abs(
                float(terminal_columns["collision_probability"])
                - float(collision_check["collision_mean"])
            ),
        }
    )

    diagonal = kicked_ising_circuit(n_qubits, periods=3, model="commuting")
    diagonal_z = exact_sparse_task_p2(n_qubits, diagonal, local_z_labels(n_qubits), max_terms=50_000)
    checks.append(
        {
            "case": "commuting_kicked_ising_local_z",
            "observed": diagonal_z,
            "expected": 0.0,
            "abs_error": abs(diagonal_z),
        }
    )

    max_error = max(float(item["abs_error"]) for item in checks)
    return {"checks": checks, "max_abs_error": max_error, "passed": max_error < 1e-10}


def run_kicked_ising() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED + 5)
    n_qubits = 8
    periods = [0, 1, 2, 3, 4]
    models = ["commuting", "scrambling"]
    tasks: list[tuple[str, Callable[[int, random.Random, int], list[Label]]]] = [
        ("local_z", lambda n, r, s: local_z_labels(n)),
        ("local_transverse", lambda n, r, s: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r, s: hamiltonian_local_labels(n)),
        ("all_pauli", sample_all_labels),
    ]
    rows: list[dict[str, object]] = []
    for model in models:
        for period in periods:
            gates = kicked_ising_circuit(n_qubits, period, model)
            for task_name, label_fn in tasks:
                label_rng = random.Random(rng.randrange(1 << 60))
                labels = label_fn(n_qubits, label_rng, 24)
                terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=120_000)
                profile = terminal_truncation_profile(terminal)
                rows.append(
                    {
                        "model": model,
                        "n_qubits": n_qubits,
                        "periods": period,
                        "task": task_name,
                        "task_p2": profile["task_p2"],
                        "exp_task_p2": profile["exp_task_p2"],
                        "k90": profile["k90"],
                        "terminal_support": profile["terminal_support"],
                        "collision_mean": terminal["collision_probability"],
                        "labels_requested": terminal["labels_requested"],
                        "labels_used": terminal["labels_used"],
                        "capped_labels": terminal["capped_labels"],
                        "avg_terminal_terms": terminal["avg_terminal_terms"],
                        "max_terminal_terms": terminal["max_terminal_terms"],
                    }
                )
    return rows


def kicked_ising_circuit(n_qubits: int, periods: int, model: str) -> list[Gate]:
    gates: list[Gate] = []
    for _ in range(periods):
        if model == "commuting":
            for q in range(n_qubits):
                gates.append(gate("T", q))
            for q in range(n_qubits - 1):
                gates.append(gate("CZ", q, q + 1))
        elif model == "scrambling":
            for q in range(0, n_qubits - 1, 2):
                gates.append(gate("CZ", q, q + 1))
            for q in range(1, n_qubits - 1, 2):
                gates.append(gate("CZ", q, q + 1))
            for q in range(n_qubits):
                gates.append(gate("T", q))
            for q in range(n_qubits):
                gates.append(gate("H", q))
        else:
            raise ValueError(f"unknown kicked-Ising model: {model}")
    return gates


def run_task_scaling() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED + 10)
    configs = [
        {"n_qubits": 8, "layers": 5, "samples": 24, "repeats": 20},
        {"n_qubits": 10, "layers": 5, "samples": 22, "repeats": 20},
        {"n_qubits": 12, "layers": 4, "samples": 20, "repeats": 20},
    ]
    densities = [0.0, 0.10, 0.20, 0.30]
    tasks: list[tuple[str, Callable[[int, random.Random, int], list[Label]]]] = [
        ("local_z", lambda n, r, s: local_z_labels(n)),
        ("local_transverse", lambda n, r, s: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r, s: hamiltonian_local_labels(n)),
        ("random_local", sample_random_local_labels),
        ("transported_local_z", transported_local_z_labels),
        ("full_transverse", sample_full_transverse_labels),
        ("all_pauli", sample_all_labels),
    ]
    rows: list[dict[str, object]] = []
    for config in configs:
        n_qubits = int(config["n_qubits"])
        layers = int(config["layers"])
        samples = int(config["samples"])
        repeats = int(config["repeats"])
        for density in densities:
            by_task: dict[str, list[dict[str, object]]] = {name: [] for name, _ in tasks}
            for _ in range(repeats):
                circuit_rng = random.Random(rng.randrange(1 << 60))
                gates = random_clifford_t_circuit(n_qubits, layers, density, circuit_rng)
                for task_name, label_fn in tasks:
                    label_rng = random.Random(rng.randrange(1 << 60))
                    labels = label_fn(n_qubits, label_rng, samples)
                    by_task[task_name].append(
                        estimate_task_p2(
                            n_qubits,
                            gates,
                            labels,
                            max_terms=120_000,
                        )
                    )
            for task_name, estimates in by_task.items():
                rows.append(summarize_task(n_qubits, layers, density, task_name, estimates))
    return rows


def run_global_task_truncation() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED + 15)
    configs = [
        {"n_qubits": 8, "layers": 5, "samples": 24},
        {"n_qubits": 10, "layers": 5, "samples": 22},
        {"n_qubits": 12, "layers": 4, "samples": 20},
    ]
    densities = [0.10, 0.20, 0.30]
    repeats = 20
    tasks: list[tuple[str, Callable[[int, random.Random, int], list[Label]]]] = [
        ("local_z", lambda n, r, s: local_z_labels(n)),
        ("local_transverse", lambda n, r, s: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r, s: hamiltonian_local_labels(n)),
        ("transported_local_z", transported_local_z_labels),
        ("all_pauli", sample_all_labels),
    ]
    rows: list[dict[str, object]] = []
    for config in configs:
        n_qubits = int(config["n_qubits"])
        layers = int(config["layers"])
        samples = int(config["samples"])
        for density in densities:
            for repeat in range(repeats):
                circuit_seed = rng.randrange(1 << 60)
                gates = random_clifford_t_circuit(
                    n_qubits,
                    layers,
                    density,
                    random.Random(circuit_seed),
                )
                global_labels = sample_all_labels(
                    n_qubits,
                    random.Random(rng.randrange(1 << 60)),
                    max(32, samples),
                )
                global_terminal = build_terminal_columns(
                    n_qubits,
                    gates,
                    global_labels,
                    max_terms=120_000,
                )
                global_profile = terminal_truncation_profile(global_terminal)
                for task_name, label_fn in tasks:
                    labels = label_fn(
                        n_qubits,
                        random.Random(rng.randrange(1 << 60)),
                        samples,
                    )
                    terminal = build_terminal_columns(
                        n_qubits,
                        gates,
                        labels,
                        max_terms=120_000,
                    )
                    profile = terminal_truncation_profile(terminal)
                    rows.append(
                        {
                            "n_qubits": n_qubits,
                            "layers": layers,
                            "t_density": density,
                            "expected_t_gates": n_qubits * layers * density,
                            "repeat": repeat,
                            "circuit_seed": circuit_seed,
                            "task": task_name,
                            "global_p2_estimate": global_profile["task_p2"],
                            "global_k90_estimate": global_profile["k90"],
                            "task_p2": profile["task_p2"],
                            "exp_task_p2": profile["exp_task_p2"],
                            "k50": profile["k50"],
                            "k90": profile["k90"],
                            "k95": profile["k95"],
                            "k90_lower_bound": 0.81 * profile["exp_task_p2"],
                            "k90_over_exp_p2": profile["k90"] / profile["exp_task_p2"],
                            "terminal_support": profile["terminal_support"],
                            "labels_requested": terminal["labels_requested"],
                            "labels_used": terminal["labels_used"],
                            "capped_labels": terminal["capped_labels"],
                            "avg_terminal_terms": terminal["avg_terminal_terms"],
                        }
                    )
    return rows


def run_error_collapse() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    rng = random.Random(RNG_SEED + 20)
    cases = [
        {
            "case": "local Z n8",
            "n_qubits": 8,
            "layers": 5,
            "density": 0.25,
            "labels": lambda n, r, s: local_z_labels(n),
        },
        {
            "case": "local X/Y n8",
            "n_qubits": 8,
            "layers": 5,
            "density": 0.25,
            "labels": lambda n, r, s: local_transverse_labels(n),
        },
        {
            "case": "Hamiltonian n10",
            "n_qubits": 10,
            "layers": 4,
            "density": 0.25,
            "labels": lambda n, r, s: hamiltonian_local_labels(n),
        },
        {
            "case": "transported Z n12",
            "n_qubits": 12,
            "layers": 4,
            "density": 0.20,
            "labels": transported_local_z_labels,
        },
        {
            "case": "all Pauli n12",
            "n_qubits": 12,
            "layers": 4,
            "density": 0.20,
            "labels": sample_all_labels,
        },
    ]
    sample_counts = [128, 256, 512, 1024, 2048]
    trials = 100
    rows: list[dict[str, object]] = []
    truncation_rows: list[dict[str, object]] = []
    observable_error_rows: list[dict[str, object]] = []
    for case in cases:
        n_qubits = int(case["n_qubits"])
        circuit_rng = random.Random(rng.randrange(1 << 60))
        gates = random_clifford_t_circuit(
            n_qubits,
            layers=int(case["layers"]),
            t_density=float(case["density"]),
            rng=circuit_rng,
        )
        label_rng = random.Random(rng.randrange(1 << 60))
        labels = case["labels"](n_qubits, label_rng, 24)
        terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=120_000)
        truncation_rows.append(
            {
                "case": case["case"],
                "n_qubits": n_qubits,
                "layers": int(case["layers"]),
                "t_density": float(case["density"]),
                **terminal_truncation_profile(terminal),
            }
        )
        observable_error_rows.extend(
            terminal_observable_truncation_errors(
                terminal,
                {
                    "case": case["case"],
                    "n_qubits": n_qubits,
                    "layers": int(case["layers"]),
                    "t_density": float(case["density"]),
                },
            )
        )
        p = float(terminal["collision_probability"])
        p2 = -math.log(max(p, 1e-300))
        columns = terminal["columns"]
        for samples in sample_counts:
            estimates = []
            for _ in range(trials):
                estimates.append(run_terminal_collision_estimator(columns, samples, rng))
            estimate_array = np.asarray(estimates, dtype=float)
            rmse = float(np.sqrt(np.mean((estimate_array - p) ** 2)))
            relative_rmse = rmse / max(p, 1e-300)
            rows.append(
                {
                    "case": case["case"],
                    "n_qubits": n_qubits,
                    "layers": int(case["layers"]),
                    "t_density": float(case["density"]),
                    "task_p2": p2,
                    "collision_probability": p,
                    "samples": samples,
                    "effective_samples": samples * p,
                    "trials": trials,
                    "relative_rmse": relative_rmse,
                    "theory_relative_rmse": math.sqrt((1.0 - p) / max(samples * p, 1e-300)),
                    "labels_used": terminal["labels_used"],
                    "capped_labels": terminal["capped_labels"],
                    "avg_terminal_terms": terminal["avg_terminal_terms"],
                }
            )
    return rows, truncation_rows, observable_error_rows


def build_terminal_columns(
    n_qubits: int,
    gates: Iterable[Gate],
    labels: list[Label],
    max_terms: int,
) -> dict[str, object]:
    gate_list = list(gates)
    columns: list[np.ndarray] = []
    probability_maps: list[dict[Label, float]] = []
    coefficient_maps: list[dict[Label, float]] = []
    collisions: list[float] = []
    term_counts: list[int] = []
    capped = 0
    for label in labels:
        try:
            coeffs, _ = propagate_sparse_pauli(n_qubits, gate_list, label, max_terms=max_terms)
        except RuntimeError:
            capped += 1
            continue
        probabilities = np.asarray([value * value for value in coeffs.values()], dtype=float)
        probabilities /= float(np.sum(probabilities))
        columns.append(np.cumsum(probabilities))
        probability_maps.append(
            {label: float(value * value) for label, value in coeffs.items()}
        )
        coefficient_maps.append({label: float(value) for label, value in coeffs.items()})
        collisions.append(float(np.sum(probabilities**2)))
        term_counts.append(len(probabilities))
    if not columns:
        raise RuntimeError("all labels exceeded the sparse propagation cap")
    return {
        "columns": columns,
        "probability_maps": probability_maps,
        "coefficient_maps": coefficient_maps,
        "collision_probability": float(np.mean(collisions)),
        "labels_requested": len(labels),
        "labels_used": len(columns),
        "capped_labels": capped,
        "avg_terminal_terms": float(np.mean(term_counts)),
        "max_terminal_terms": int(max(term_counts)),
    }


def terminal_truncation_profile(terminal: dict[str, object]) -> dict[str, object]:
    maps = terminal["probability_maps"]
    labels_used = int(terminal["labels_used"])
    marginal: dict[Label, float] = {}
    for item in maps:
        for label, probability in item.items():
            marginal[label] = marginal.get(label, 0.0) + float(probability) / labels_used
    sorted_mass = sorted(marginal.values(), reverse=True)
    marginal_collision = sum(value * value for value in sorted_mass)
    collision_probability = float(terminal["collision_probability"])
    return {
        "task_p2": -math.log(max(collision_probability, 1e-300)),
        "exp_task_p2": 1.0 / max(collision_probability, 1e-300),
        "collision_probability": collision_probability,
        "marginal_collision": marginal_collision,
        "marginal_effective_support": 1.0 / max(marginal_collision, 1e-300),
        "terminal_support": len(sorted_mass),
        "k50": mass_cutoff(sorted_mass, 0.50),
        "k90": mass_cutoff(sorted_mass, 0.90),
        "k95": mass_cutoff(sorted_mass, 0.95),
        "labels_used": labels_used,
        "capped_labels": int(terminal["capped_labels"]),
        "avg_terminal_terms": float(terminal["avg_terminal_terms"]),
    }


def terminal_observable_truncation_errors(
    terminal: dict[str, object],
    metadata: dict[str, object],
    targets: list[float] | None = None,
) -> list[dict[str, object]]:
    coefficient_maps = terminal["coefficient_maps"]
    n_qubits = int(metadata["n_qubits"])
    if targets is None:
        targets = [0.50, 0.70, 0.90, 0.95]
    states: list[tuple[str, Callable[[Label], float]]] = [
        ("z_product", lambda label: 1.0 if label[0] == 0 else 0.0),
        ("x_product", lambda label: 1.0 if label[1] == 0 else 0.0),
        ("tilted_product", lambda label: tilted_product_response(n_qubits, label)),
    ]
    rows: list[dict[str, object]] = []
    for state_name, response in states:
        exact_values = np.asarray(
            [
                sum(coeff * response(label) for label, coeff in item.items())
                for item in coefficient_maps
            ],
            dtype=float,
        )
        exact_rms = float(np.sqrt(np.mean(exact_values**2)))
        normalizer = max(exact_rms, 1e-12)
        for target in targets:
            errors: list[float] = []
            retained_terms: list[int] = []
            retained_masses: list[float] = []
            for item, exact in zip(coefficient_maps, exact_values):
                sorted_items = sorted(item.items(), key=lambda pair: pair[1] * pair[1], reverse=True)
                total = 0.0
                retained: list[tuple[Label, float]] = []
                for label, coeff in sorted_items:
                    retained.append((label, coeff))
                    total += coeff * coeff
                    if total >= target:
                        break
                approx = sum(coeff * response(label) for label, coeff in retained)
                errors.append(approx - float(exact))
                retained_terms.append(len(retained))
                retained_masses.append(total)
            rms_error = float(np.sqrt(np.mean(np.asarray(errors, dtype=float) ** 2)))
            rows.append(
                {
                    **metadata,
                    "state_response": state_name,
                    "target_mass": target,
                    "mean_retained_mass": float(np.mean(retained_masses)),
                    "avg_retained_terms": float(np.mean(retained_terms)),
                    "rms_error": rms_error,
                    "relative_rms_error": rms_error / normalizer,
                    "exact_rms": exact_rms,
                    "task_p2": -math.log(
                        max(float(terminal["collision_probability"]), 1e-300)
                    ),
                    "labels_used": int(terminal["labels_used"]),
                    "capped_labels": int(terminal["capped_labels"]),
                }
            )
    return rows


def tilted_product_response(n_qubits: int, label: Label) -> float:
    xmask, zmask = label
    theta = 0.73
    phi = 0.41
    bx = math.sin(theta) * math.cos(phi)
    by = math.sin(theta) * math.sin(phi)
    bz = math.cos(theta)
    value = 1.0
    for q in range(n_qubits):
        x_bit = bool(xmask & (1 << q))
        z_bit = bool(zmask & (1 << q))
        if x_bit and z_bit:
            value *= by
        elif x_bit:
            value *= bx
        elif z_bit:
            value *= bz
    return value


def mass_cutoff(sorted_mass: list[float], target: float) -> int:
    total = 0.0
    for index, value in enumerate(sorted_mass, start=1):
        total += value
        if total >= target:
            return index
    return len(sorted_mass)


def run_terminal_collision_estimator(
    columns: list[np.ndarray],
    samples: int,
    rng: random.Random,
) -> float:
    hits = 0
    for _ in range(samples):
        cumulative = columns[rng.randrange(len(columns))]
        first = int(np.searchsorted(cumulative, rng.random(), side="right"))
        second = int(np.searchsorted(cumulative, rng.random(), side="right"))
        if first == second:
            hits += 1
    return hits / samples


def estimate_task_p2(
    n_qubits: int,
    gates: Iterable[Gate],
    labels: list[Label],
    max_terms: int,
) -> dict[str, object]:
    gate_list = list(gates)
    collisions: list[float] = []
    capped = 0
    term_counts: list[int] = []
    for label in labels:
        try:
            coeffs, _ = propagate_sparse_pauli(n_qubits, gate_list, label, max_terms=max_terms)
        except RuntimeError:
            capped += 1
            continue
        collisions.append(sum(value**4 for value in coeffs.values()))
        term_counts.append(len(coeffs))
    if not collisions:
        raise RuntimeError("all labels exceeded the sparse propagation cap")
    collision_array = np.asarray(collisions, dtype=float)
    mean_collision = float(np.mean(collision_array))
    return {
        "task_p2": -math.log(max(mean_collision, 1e-300)),
        "collision_mean": mean_collision,
        "collision_sem": float(np.std(collision_array, ddof=1) / math.sqrt(len(collision_array)))
        if len(collision_array) > 1
        else 0.0,
        "labels_requested": len(labels),
        "labels_used": len(collisions),
        "capped_labels": capped,
        "avg_terminal_terms": float(np.mean(term_counts)),
        "max_terminal_terms": int(max(term_counts)),
    }


def exact_sparse_task_p2(
    n_qubits: int,
    gates: Iterable[Gate],
    labels: list[Label],
    max_terms: int,
) -> float:
    return float(estimate_task_p2(n_qubits, gates, labels, max_terms=max_terms)["task_p2"])


def summarize_task(
    n_qubits: int,
    layers: int,
    density: float,
    task_name: str,
    estimates: list[dict[str, object]],
) -> dict[str, object]:
    p2_values = np.asarray([float(item["task_p2"]) for item in estimates], dtype=float)
    requested = sum(int(item["labels_requested"]) for item in estimates)
    used = sum(int(item["labels_used"]) for item in estimates)
    capped = sum(int(item["capped_labels"]) for item in estimates)
    return {
        "n_qubits": n_qubits,
        "layers": layers,
        "t_density": density,
        "expected_t_gates": n_qubits * layers * density,
        "task": task_name,
        "task_p2_mean": float(np.mean(p2_values)),
        "task_p2_sem_over_circuits": float(np.std(p2_values, ddof=1) / math.sqrt(len(p2_values)))
        if len(p2_values) > 1
        else 0.0,
        "labels_requested": requested,
        "labels_used": used,
        "capped_labels": capped,
        "avg_terminal_terms": float(np.mean([float(item["avg_terminal_terms"]) for item in estimates])),
        "max_terminal_terms": int(max(int(item["max_terminal_terms"]) for item in estimates)),
    }


def all_labels(n_qubits: int) -> list[Label]:
    return [(xmask, zmask) for xmask in range(1 << n_qubits) for zmask in range(1 << n_qubits)]


def local_z_labels(n_qubits: int) -> list[Label]:
    return [(0, 1 << q) for q in range(n_qubits)]


def local_transverse_labels(n_qubits: int) -> list[Label]:
    labels: list[Label] = []
    for q in range(n_qubits):
        labels.append((1 << q, 0))
        labels.append((1 << q, 1 << q))
    return labels


def full_transverse_labels(n_qubits: int) -> list[Label]:
    xmask = (1 << n_qubits) - 1
    return [(xmask, zmask) for zmask in range(1 << n_qubits)]


def hamiltonian_local_labels(n_qubits: int) -> list[Label]:
    labels: list[Label] = []
    for q in range(n_qubits - 1):
        pair = (1 << q) | (1 << (q + 1))
        labels.append((pair, 0))
        labels.append((pair, pair))
        labels.append((0, pair))
    return labels


def sample_all_labels(n_qubits: int, rng: random.Random, samples: int) -> list[Label]:
    return [(rng.randrange(1 << n_qubits), rng.randrange(1 << n_qubits)) for _ in range(samples)]


def sample_full_transverse_labels(n_qubits: int, rng: random.Random, samples: int) -> list[Label]:
    xmask = (1 << n_qubits) - 1
    return [(xmask, rng.randrange(1 << n_qubits)) for _ in range(samples)]


def sample_random_local_labels(n_qubits: int, rng: random.Random, samples: int) -> list[Label]:
    labels: list[Label] = []
    for _ in range(samples):
        support_size = 1 if rng.random() < 0.5 or n_qubits == 1 else 2
        qubits = rng.sample(range(n_qubits), support_size)
        xmask = 0
        zmask = 0
        for q in qubits:
            code = rng.choice([1, 2, 3])
            if code in {1, 2}:
                xmask |= 1 << q
            if code in {2, 3}:
                zmask |= 1 << q
        labels.append((xmask, zmask))
    return labels


def transported_local_z_labels(n_qubits: int, rng: random.Random, samples: int) -> list[Label]:
    frame = random_clifford_t_circuit(n_qubits, layers=2, t_density=0.0, rng=rng)
    return [transport_label(n_qubits, label, frame) for label in local_z_labels(n_qubits)]


def transport_label(n_qubits: int, label: Label, clifford_gates: Iterable[Gate]) -> Label:
    transported = label
    for item in reversed(list(clifford_gates)):
        transported = _apply_clifford_label(n_qubits, transported, item)
    return transported


def label_to_index(label: Label) -> int:
    xmask, zmask = label
    index = 0
    q = 0
    while (1 << q) <= max(xmask, zmask):
        x_bit = bool(xmask & (1 << q))
        z_bit = bool(zmask & (1 << q))
        if x_bit and z_bit:
            code = 2
        elif x_bit:
            code = 1
        elif z_bit:
            code = 3
        else:
            code = 0
        index += code * (4**q)
        q += 1
    return index


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
