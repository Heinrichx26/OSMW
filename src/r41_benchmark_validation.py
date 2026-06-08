from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from r41_observable_cost import (
    Label,
    build_terminal_columns,
    hamiltonian_local_labels,
    local_transverse_labels,
    local_z_labels,
    sample_full_transverse_labels,
    sample_all_labels,
    sample_random_local_labels,
    terminal_truncation_profile,
    transported_local_z_labels,
)
from r41_path_entropy import Gate, gate, pauli_transfer_matrix, unitary_from_gates
from r41_task_resolved_checks import task_entropy_from_transfer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RNG_SEED = 41061
SUPPORTED_GATES = {"h", "s", "sdg", "t", "tdg", "x", "z", "cx", "cz"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    validation = run_smoke()
    (DATA_DIR / "r41_benchmark_validation_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"validation": validation}, indent=2))
    if args.validate or args.smoke:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    rows, summary = run_benchmark_validation()
    write_csv(DATA_DIR / "r41_benchmark_validation.csv", rows)
    write_csv(DATA_DIR / "r41_benchmark_validation_summary.csv", summary)
    (DATA_DIR / "r41_benchmark_validation.json").write_text(
        json.dumps({"validation": validation, "rows": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )


def run_smoke() -> dict[str, object]:
    circuit = load_mqt_circuit("grover", 3)
    gates = qiskit_to_gates(circuit)
    transfer = pauli_transfer_matrix(unitary_from_gates(circuit.num_qubits, gates))
    labels = local_transverse_labels(circuit.num_qubits)
    exact = task_entropy_from_transfer(transfer, [label_to_index(label) for label in labels])
    terminal = build_terminal_columns(circuit.num_qubits, gates, labels, max_terms=80_000)
    sparse = float(terminal_truncation_profile(terminal)["task_p2"])
    error = abs(exact - sparse)
    return {
        "benchmark": "grover",
        "n_qubits": circuit.num_qubits,
        "task": "local_transverse",
        "exact": exact,
        "sparse": sparse,
        "abs_error": error,
        "passed": error < 1e-10,
    }


def run_benchmark_validation() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(RNG_SEED)
    rows: list[dict[str, object]] = []
    specs = [
        ("bv", 3),
        ("bv", 5),
        ("bv", 7),
        ("ghz", 3),
        ("ghz", 5),
        ("ghz", 7),
        ("graphstate", 5),
        ("graphstate", 7),
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
        ("qft", 2),
        ("qftentangled", 2),
        ("qpeexact", 3),
        ("qpeinexact", 2),
        ("wstate", 2),
    ]
    for benchmark, size in specs:
        try:
            circuit = load_mqt_circuit(benchmark, size)
            gates = qiskit_to_gates(circuit)
        except Exception as exc:
            rows.append(skip_row(benchmark, size, f"load:{type(exc).__name__}"))
            continue
        t_count = sum(1 for item in gates if item.name in {"T", "TDG"})
        if t_count > 32:
            rows.append(skip_row(benchmark, size, "t_count_filter", circuit.num_qubits, t_count))
            continue
        task_fns = [
            ("local_z", lambda n, r: local_z_labels(n)),
            ("local_x", lambda n, r: local_axis_labels(n, "x")),
            ("local_y", lambda n, r: local_axis_labels(n, "y")),
            ("local_transverse", lambda n, r: local_transverse_labels(n)),
            ("pair_xx", lambda n, r: pair_axis_labels(n, "xx")),
            ("pair_zz", lambda n, r: pair_axis_labels(n, "zz")),
            ("hamiltonian_terms", lambda n, r: hamiltonian_local_labels(n)),
            ("random_local_a", lambda n, r: sample_random_local_labels(n, r, min(32, 4 * n))),
            ("random_local_b", lambda n, r: sample_random_local_labels(n, r, min(32, 4 * n))),
            ("transported_local_z", lambda n, r: transported_local_z_labels(n, r, min(32, 4 * n))),
            ("sampled_transverse", lambda n, r: sample_full_transverse_labels(n, r, min(64, 4**n))),
            ("all_pauli", lambda n, r: sample_all_labels(n, r, min(96, 4**n))),
        ]
        all_terminal = None
        try:
            all_terminal = build_terminal_columns(
                circuit.num_qubits,
                gates,
                sample_all_labels(circuit.num_qubits, random.Random(rng.randrange(1 << 60)), 96),
                max_terms=200_000,
            )
            all_profile = terminal_truncation_profile(all_terminal)
        except RuntimeError:
            all_profile = {
                "task_p2": math.nan,
                "k90": math.nan,
            }
        for task_name, label_fn in task_fns:
            label_rng = random.Random(rng.randrange(1 << 60))
            labels = label_fn(circuit.num_qubits, label_rng)
            try:
                terminal = build_terminal_columns(
                    circuit.num_qubits,
                    gates,
                    labels,
                    max_terms=200_000,
                )
                profile = terminal_truncation_profile(terminal)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "size": size,
                        "n_qubits": circuit.num_qubits,
                        "gates": len(gates),
                        "t_count": t_count,
                        "task": task_name,
                        "task_p2": profile["task_p2"],
                        "exp_task_p2": profile["exp_task_p2"],
                        "k90": profile["k90"],
                        "k95": profile["k95"],
                        "gamma90": profile["k90"] / (0.81 * profile["exp_task_p2"]),
                        "input_mixing_gap": profile["collision_probability"] / max(profile["marginal_collision"], 1e-300),
                        "terminal_tail_gap": profile["k90"] * profile["marginal_collision"] / 0.81,
                        "terminal_support": profile["terminal_support"],
                        "global_p2_estimate": all_profile["task_p2"],
                        "global_k90_estimate": all_profile["k90"],
                        "labels_used": terminal["labels_used"],
                        "capped_labels": terminal["capped_labels"],
                        "avg_terminal_terms": terminal["avg_terminal_terms"],
                        "status": "ok",
                    }
                )
            except RuntimeError:
                rows.append(
                    skip_row(
                        benchmark,
                        size,
                        "sparse_cap",
                        circuit.num_qubits,
                        t_count,
                        task_name,
                        len(gates),
                    )
                )
    valid_rows = [row for row in rows if row.get("status") == "ok" and row["k90"] > 0]
    summary = predictor_summary(valid_rows)
    return rows, summary


def load_mqt_circuit(benchmark: str, size: int):
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qiskit import transpile

    circuit = get_benchmark(
        benchmark,
        BenchmarkLevel.ALG,
        circuit_size=size,
        random_parameters=False,
    )
    return transpile(
        circuit,
        basis_gates=sorted(SUPPORTED_GATES),
        optimization_level=0,
    )


def qiskit_to_gates(circuit) -> list[Gate]:
    qubit_index = {bit: index for index, bit in enumerate(circuit.qubits)}
    gates: list[Gate] = []
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name.lower()
        if name in {"measure", "barrier", "delay"}:
            continue
        if name not in SUPPORTED_GATES:
            raise ValueError(f"unsupported transpiled gate: {name}")
        qubits = tuple(qubit_index[qubit] for qubit in instruction.qubits)
        if name == "cx":
            gates.append(gate("CNOT", *qubits))
        else:
            gates.append(gate(name.upper(), *qubits))
    return gates


def predictor_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    output = []
    subsets = [
        ("mqt_bench", rows),
        ("mqt_bench_nonclifford", [row for row in rows if int(row["t_count"]) > 0]),
        (
            "mqt_bench_nonclifford_core",
            [
                row
                for row in rows
                if int(row["t_count"]) > 0
                and row["task"]
                in {"local_z", "local_transverse", "hamiltonian_terms", "all_pauli"}
            ],
        ),
        (
            "mqt_bench_nonclifford_local",
            [
                row
                for row in rows
                if int(row["t_count"]) > 0 and row["task"] != "all_pauli"
            ],
        ),
    ]
    for subset_name, subset_rows in subsets:
        output.extend(predictor_fit_rows(subset_name, subset_rows))
        output.extend(ratio_summary_rows(subset_name, subset_rows))
        output.extend(tail_factor_summary_rows(subset_name, subset_rows))
    return output


def predictor_fit_rows(
    subset_name: str,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(rows) < 2:
        return []
    output = []
    for predictor, field in [
        ("global_operator_sre", "global_p2_estimate"),
        ("task_entropy", "task_p2"),
    ]:
        filtered = [
            (float(row[field]), math.log(float(row["k90"])))
            for row in rows
            if math.isfinite(float(row[field])) and float(row["k90"]) > 0
        ]
        if len(filtered) < 2:
            continue
        fit = linear_fit([item[0] for item in filtered], [item[1] for item in filtered])
        output.append(
            {
                "subset": subset_name,
                "predictor": predictor,
                "pairs": len(filtered),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "coefficient_of_determination": fit["r2"],
                "pearson_correlation": fit["pearson"],
            }
        )
    return output


def ratio_summary_rows(
    subset_name: str,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    task_rows = [row for row in rows if row["task"] != "all_pauli"]
    all_by_benchmark = {
        (row["benchmark"], row["size"]): float(row["k90"])
        for row in rows
        if row["task"] == "all_pauli"
    }
    ratios = [
        float(row["k90"]) / all_by_benchmark[(row["benchmark"], row["size"])]
        for row in task_rows
        if (row["benchmark"], row["size"]) in all_by_benchmark
        and all_by_benchmark[(row["benchmark"], row["size"])] > 0
    ]
    if ratios:
        return [
            {
                "subset": subset_name,
                "predictor": "same_circuit_task_ratio",
                "pairs": len(ratios),
                "slope": math.nan,
                "intercept": float(np.median(ratios)),
                "coefficient_of_determination": math.nan,
                "pearson_correlation": math.nan,
            }
        ]
    return []


def tail_factor_summary_rows(
    subset_name: str,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for name, field in [
        ("median_gamma90", "gamma90"),
        ("median_input_mixing_gap", "input_mixing_gap"),
        ("median_terminal_tail_gap", "terminal_tail_gap"),
    ]:
        values = [
            float(row[field])
            for row in rows
            if field in row and math.isfinite(float(row[field])) and float(row[field]) > 0
        ]
        if values:
            output.append(
                {
                    "subset": subset_name,
                    "predictor": name,
                    "pairs": len(values),
                    "slope": math.nan,
                    "intercept": float(np.median(values)),
                    "coefficient_of_determination": math.nan,
                    "pearson_correlation": math.nan,
                }
            )
    return output


def linear_fit(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    design = np.vstack([np.ones(len(x)), x]).T
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 0.0
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "pearson": pearson,
    }


def skip_row(
    benchmark: str,
    size: int,
    status: str,
    n_qubits: int | None = None,
    t_count: int | None = None,
    task: str = "",
    gates_count: int | None = None,
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "size": size,
        "n_qubits": n_qubits if n_qubits is not None else -1,
        "gates": gates_count if gates_count is not None else -1,
        "t_count": t_count if t_count is not None else -1,
        "task": task,
        "task_p2": math.nan,
        "exp_task_p2": math.nan,
        "k90": -1,
        "k95": -1,
        "gamma90": math.nan,
        "input_mixing_gap": math.nan,
        "terminal_tail_gap": math.nan,
        "terminal_support": -1,
        "global_p2_estimate": math.nan,
        "global_k90_estimate": -1,
        "labels_used": 0,
        "capped_labels": 0,
        "avg_terminal_terms": math.nan,
        "status": status,
    }


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


def local_axis_labels(n_qubits: int, axis: str) -> list[Label]:
    labels: list[Label] = []
    for q in range(n_qubits):
        if axis == "x":
            labels.append((1 << q, 0))
        elif axis == "y":
            labels.append((1 << q, 1 << q))
        elif axis == "z":
            labels.append((0, 1 << q))
        else:
            raise ValueError(f"unknown axis: {axis}")
    return labels


def pair_axis_labels(n_qubits: int, axis: str) -> list[Label]:
    labels: list[Label] = []
    for q in range(n_qubits - 1):
        pair = (1 << q) | (1 << (q + 1))
        if axis == "xx":
            labels.append((pair, 0))
        elif axis == "yy":
            labels.append((pair, pair))
        elif axis == "zz":
            labels.append((0, pair))
        else:
            raise ValueError(f"unknown pair axis: {axis}")
    return labels


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
