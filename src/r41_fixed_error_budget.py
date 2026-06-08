from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Callable

from r41_benchmark_validation import load_mqt_circuit, qiskit_to_gates
from r41_observable_cost import (
    Label,
    build_terminal_columns,
    hamiltonian_local_labels,
    kicked_ising_circuit,
    local_transverse_labels,
    local_z_labels,
    random_clifford_t_circuit,
    sample_all_labels,
    terminal_observable_truncation_errors,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RNG_SEED = 41079
ERROR_TARGETS = [0.50, 0.70, 0.90, 0.95, 0.98, 0.99, 1.00]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    validation = run_validation()
    (DATA_DIR / "r41_fixed_error_budget_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.validate:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    rows = run_fixed_error_budget()
    summary = summarize(rows)
    write_csv(DATA_DIR / "r41_fixed_error_budget.csv", rows)
    write_csv(DATA_DIR / "r41_fixed_error_budget_summary.csv", summary)
    (DATA_DIR / "r41_fixed_error_budget.json").write_text(
        json.dumps({"validation": validation, "rows": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )


def run_validation() -> dict[str, object]:
    rng = random.Random(RNG_SEED)
    n_qubits = 6
    gates = random_clifford_t_circuit(n_qubits, layers=3, t_density=0.20, rng=rng)
    labels = hamiltonian_local_labels(n_qubits)
    terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=50_000)
    rows = terminal_observable_truncation_errors(
        terminal,
        {"case": "validation", "n_qubits": n_qubits, "layers": 3, "t_density": 0.20},
    )
    tilted = [row for row in rows if row["state_response"] == "tilted_product"]
    monotone_terms = all(
        float(left["avg_retained_terms"]) <= float(right["avg_retained_terms"])
        for left, right in zip(tilted, tilted[1:])
    )
    return {
        "case": "random_hamiltonian_validation",
        "n_qubits": n_qubits,
        "labels_used": terminal["labels_used"],
        "max_relative_rmse": max(float(row["relative_rms_error"]) for row in tilted),
        "monotone_retained_terms": monotone_terms,
        "passed": bool(tilted) and monotone_terms,
    }


def run_fixed_error_budget() -> list[dict[str, object]]:
    rng = random.Random(RNG_SEED + 1)
    rows: list[dict[str, object]] = []
    rows.extend(run_random_budget(rng))
    rows.extend(run_floquet_budget(rng))
    rows.extend(run_mqt_budget(rng))
    return rows


def run_random_budget(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_qubits = 10
    layers = 4
    density = 0.20
    tasks: list[tuple[str, Callable[[int, random.Random], list[Label]]]] = [
        ("local_z", lambda n, r: local_z_labels(n)),
        ("local_transverse", lambda n, r: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r: hamiltonian_local_labels(n)),
        ("all_pauli", lambda n, r: sample_all_labels(n, r, 24)),
    ]
    for repeat in range(20):
        circuit_seed = rng.randrange(1 << 60)
        gates = random_clifford_t_circuit(
            n_qubits,
            layers=layers,
            t_density=density,
            rng=random.Random(circuit_seed),
        )
        circuit_id = f"random_n{n_qubits}_r{repeat}"
        for task, label_fn in tasks:
            label_rng = task_rng(circuit_seed, task)
            labels = label_fn(n_qubits, label_rng)
            rows.append(
                evaluate_task(
                    dataset="random",
                    circuit_id=circuit_id,
                    task=task,
                    n_qubits=n_qubits,
                    gates=gates,
                    labels=labels,
                    threshold=0.10,
                    metadata={
                        "layers": layers,
                        "t_density": density,
                        "repeat": repeat,
                        "circuit_seed": circuit_seed,
                    },
                )
            )
    return rows


def run_mqt_budget(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = [
        ("bv", 3),
        ("bv", 5),
        ("bv", 7),
        ("dj", 5),
        ("dj", 7),
        ("grover", 3),
        ("half_adder", 3),
        ("half_adder", 5),
        ("full_adder", 4),
        ("full_adder", 6),
        ("modular_adder", 4),
        ("modular_adder", 6),
        ("modular_adder", 8),
        ("modular_adder", 10),
        ("cdkm_ripple_carry_adder", 4),
        ("cdkm_ripple_carry_adder", 6),
        ("vbe_ripple_carry_adder", 4),
        ("draper_qft_adder", 4),
    ]
    tasks: list[tuple[str, Callable[[int, random.Random], list[Label]]]] = [
        ("local_z", lambda n, r: local_z_labels(n)),
        ("local_transverse", lambda n, r: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r: hamiltonian_local_labels(n)),
        ("all_pauli", lambda n, r: sample_all_labels(n, r, min(96, 4**n))),
    ]
    for benchmark, size in specs:
        try:
            circuit = load_mqt_circuit(benchmark, size)
            gates = qiskit_to_gates(circuit)
        except Exception as exc:
            rows.append(skip_row("mqt", f"{benchmark}_{size}", "load", str(exc)))
            continue
        t_count = sum(1 for item in gates if item.name in {"T", "TDG"})
        if t_count <= 0 or t_count > 32:
            rows.append(skip_row("mqt", f"{benchmark}_{size}", "t_count", str(t_count)))
            continue
        circuit_id = f"{benchmark}_{size}"
        for task, label_fn in tasks:
            label_rng = task_rng(stable_seed(benchmark, size), task)
            labels = label_fn(circuit.num_qubits, label_rng)
            rows.append(
                evaluate_task(
                    dataset="mqt_core",
                    circuit_id=circuit_id,
                    task=task,
                    n_qubits=circuit.num_qubits,
                    gates=gates,
                    labels=labels,
                    threshold=0.10,
                    metadata={
                        "benchmark": benchmark,
                        "size": size,
                        "gates": len(gates),
                        "t_count": t_count,
                    },
                )
            )
    return rows


def run_floquet_budget(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_qubits = 8
    periods = [1, 2, 3, 4]
    frame_count = 4
    tasks: list[tuple[str, Callable[[int, random.Random], list[Label]]]] = [
        ("local_z", lambda n, r: local_z_labels(n)),
        ("local_transverse", lambda n, r: local_transverse_labels(n)),
        ("hamiltonian_terms", lambda n, r: hamiltonian_local_labels(n)),
        ("all_pauli", lambda n, r: sample_all_labels(n, r, 24)),
    ]
    for period in periods:
        base_gates = kicked_ising_circuit(n_qubits, period, "scrambling")
        for frame in range(frame_count):
            frame_seed = RNG_SEED + 20_000 + 100 * period + frame
            if frame == 0:
                gates = list(base_gates)
            else:
                gates = list(base_gates) + random_clifford_t_circuit(
                    n_qubits,
                    layers=1,
                    t_density=0.0,
                    rng=random.Random(frame_seed),
                )
            circuit_id = f"scrambling_floquet_p{period}_f{frame}"
            circuit_seed = RNG_SEED + 10_000 + 100 * period + frame
            for task, label_fn in tasks:
                labels = label_fn(n_qubits, task_rng(circuit_seed, task))
                rows.append(
                    evaluate_task(
                        dataset="floquet",
                        circuit_id=circuit_id,
                        task=task,
                        n_qubits=n_qubits,
                        gates=gates,
                        labels=labels,
                        threshold=0.10,
                        metadata={
                            "model": "scrambling",
                            "periods": period,
                            "frame": frame,
                        },
                    )
                )
    return rows


def evaluate_task(
    dataset: str,
    circuit_id: str,
    task: str,
    n_qubits: int,
    gates: list[object],
    labels: list[Label],
    threshold: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    try:
        terminal = build_terminal_columns(n_qubits, gates, labels, max_terms=200_000)
        error_rows = terminal_observable_truncation_errors(
            terminal,
            {"case": circuit_id, "n_qubits": n_qubits, "layers": metadata.get("layers", ""), "t_density": metadata.get("t_density", "")},
            targets=ERROR_TARGETS,
        )
    except RuntimeError as exc:
        return skip_row(dataset, circuit_id, "sparse_cap", str(exc), task, n_qubits)
    tilted = [
        row
        for row in error_rows
        if row["state_response"] == "tilted_product" and float(row["exact_rms"]) > 1e-12
    ]
    hit = None
    for row in sorted(tilted, key=lambda item: float(item["target_mass"])):
        if float(row["relative_rms_error"]) <= threshold:
            hit = row
            break
    best = min(tilted, key=lambda item: float(item["relative_rms_error"])) if tilted else None
    output = {
        "dataset": dataset,
        "circuit_id": circuit_id,
        "task": task,
        "n_qubits": n_qubits,
        "threshold": threshold,
        "hit": hit is not None,
        "target_mass": float(hit["target_mass"]) if hit else math.nan,
        "avg_retained_terms": float(hit["avg_retained_terms"]) if hit else math.nan,
        "relative_rms_error": float(hit["relative_rms_error"]) if hit else math.nan,
        "best_relative_rms_error": float(best["relative_rms_error"]) if best else math.nan,
        "best_avg_retained_terms": float(best["avg_retained_terms"]) if best else math.nan,
        "labels_used": terminal["labels_used"],
        "capped_labels": terminal["capped_labels"],
        "status": "ok",
    }
    output.update(metadata)
    return output


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("hit")]
    by_circuit: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in ok_rows:
        by_circuit.setdefault((str(row["dataset"]), str(row["circuit_id"])), {})[str(row["task"])] = row
    summary: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in ok_rows}):
        groups = [
            tasks
            for (group_dataset, _), tasks in by_circuit.items()
            if group_dataset == dataset and "all_pauli" in tasks
        ]
        for task in ["local_z", "local_transverse", "hamiltonian_terms"]:
            ratios = [
                float(tasks[task]["avg_retained_terms"]) / float(tasks["all_pauli"]["avg_retained_terms"])
                for tasks in groups
                if task in tasks and float(tasks["all_pauli"]["avg_retained_terms"]) > 0
            ]
            if ratios:
                sorted_ratios = sorted(ratios)
                summary.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "circuits": len(ratios),
                        "median_ratio": statistics.median(sorted_ratios),
                        "q25_ratio": quantile(sorted_ratios, 0.25),
                        "q75_ratio": quantile(sorted_ratios, 0.75),
                        "median_reduction_factor": statistics.median([1.0 / value for value in sorted_ratios if value > 0]),
                    }
                )
    return summary


def task_rng(seed: int, task: str) -> random.Random:
    task_offsets = {
        "local_z": 101,
        "local_transverse": 211,
        "hamiltonian_terms": 307,
        "all_pauli": 401,
    }
    return random.Random((seed + task_offsets.get(task, 0)) & ((1 << 60) - 1))


def stable_seed(*parts: object) -> int:
    value = RNG_SEED
    for part in parts:
        for char in str(part):
            value = (value * 131 + ord(char)) & ((1 << 60) - 1)
    return value


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return float(values[index])


def skip_row(
    dataset: str,
    circuit_id: str,
    reason: str,
    detail: str = "",
    task: str = "",
    n_qubits: int | str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "circuit_id": circuit_id,
        "task": task,
        "n_qubits": n_qubits,
        "threshold": 0.10,
        "hit": False,
        "target_mass": math.nan,
        "avg_retained_terms": math.nan,
        "relative_rms_error": math.nan,
        "best_relative_rms_error": math.nan,
        "best_avg_retained_terms": math.nan,
        "labels_used": 0,
        "capped_labels": 0,
        "status": reason,
        "detail": detail,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
