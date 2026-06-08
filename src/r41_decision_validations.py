"""Decision validations from stored R41 data.

The script adds two reproducibility tables:

1. Observable-blind decision validation.  A fixed channel is grouped with several
   observables.  For each group, any budget that ignores the observable has
   best possible multiplicative regret sqrt(max K / min K).  The script reports
   both endpoint groups and selected-task-only groups.
2. Task-column validation.  Stored sparse-propagation outputs are summarized to
   show how many task inputs were actually propagated.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ETA = 0.9


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except ValueError:
        return math.nan
    return value


def as_int(row: dict[str, str], key: str) -> int:
    value = as_float(row, key)
    if math.isnan(value):
        return 0
    return int(value)


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f"{value:.6e}"
    return f"{value:.6g}"


def group_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def regret_summary(groups: dict[object, list[float]]) -> dict[str, object]:
    ratios: list[float] = []
    regrets: list[float] = []
    for values in groups.values():
        clean = [value for value in values if finite_positive(value)]
        if len(clean) < 2:
            continue
        ratio = max(clean) / min(clean)
        ratios.append(ratio)
        regrets.append(math.sqrt(ratio))
    if not ratios:
        return {
            "groups": 0,
            "median_same_channel_ratio": "",
            "max_same_channel_ratio": "",
            "median_optimal_blind_regret": "",
            "max_optimal_blind_regret": "",
        }
    return {
        "groups": len(ratios),
        "median_same_channel_ratio": fmt(median(ratios)),
        "max_same_channel_ratio": fmt(max(ratios)),
        "median_optimal_blind_regret": fmt(median(regrets)),
        "max_optimal_blind_regret": fmt(max(regrets)),
    }


def make_blind_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    analytic_groups = {
        n: [1.0, float(math.ceil(ETA * (2**n)))]
        for n in (12, 24, 48, 96)
    }
    rows.append(
        {
            "source": "analytic_T_product_pair",
            **regret_summary(analytic_groups),
            "decision_object": "K_0.9 for Z_1 and X_1...X_n on identical T^n channel",
        }
    )

    large_n = group_rows(read_csv("r41_large_n_separation.csv"), ("family", "frame_depth", "n_qubits"))
    large_groups = {
        key: [as_float(row, "k90") for row in group]
        for key, group in large_n.items()
    }
    rows.append(
        {
            "source": "fixed_depth_clifford_dressed_T",
            **regret_summary(large_groups),
            "decision_object": "local Z, local X/Y, Hamiltonian, and all-Pauli tasks on one channel",
        }
    )

    random_rows = [row for row in read_csv("r41_predictor_benchmark_ratios.csv") if row.get("subset") == "all"]
    random = group_rows(random_rows, ("n_qubits", "layers", "t_density", "repeat"))
    random_groups = {
        key: [as_float(row, "k90") for row in group]
        for key, group in random.items()
    }
    rows.append(
        {
            "source": "random_clifford_T_grid",
            **regret_summary(random_groups),
            "decision_object": "same-circuit random-grid task families",
        }
    )

    floquet = group_rows(read_csv("r41_kicked_ising.csv"), ("model", "periods"))
    floquet_groups = {
        key: [as_float(row, "k90") for row in group]
        for key, group in floquet.items()
    }
    rows.append(
        {
            "source": "kicked_spin_chain",
            **regret_summary(floquet_groups),
            "decision_object": "same-dynamics Floquet task families",
        }
    )

    mqt_rows = [
        row
        for row in read_csv("r41_benchmark_validation.csv")
        if row.get("status") == "ok"
    ]
    mqt = group_rows(mqt_rows, ("benchmark", "size"))
    mqt_groups = {
        key: [as_float(row, "k90") for row in group]
        for key, group in mqt.items()
    }
    rows.append(
        {
            "source": "MQT_Bench_public_circuits",
            **regret_summary(mqt_groups),
            "decision_object": "public circuits with task family fixed after the circuit",
        }
    )

    return rows


def make_selected_blind_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    large_n = group_rows(
        [
            row
            for row in read_csv("r41_large_n_separation.csv")
            if row.get("task") != "all_pauli"
        ],
        ("family", "frame_depth", "n_qubits"),
    )
    rows.append(
        {
            "source": "fixed_depth_selected_tasks",
            **regret_summary(
                {key: [as_float(row, "k90") for row in group] for key, group in large_n.items()}
            ),
            "decision_object": "local Z, local X/Y, and Hamiltonian tasks on one channel",
        }
    )

    random_rows = [
        row
        for row in read_csv("r41_predictor_benchmark_ratios.csv")
        if row.get("subset") == "all" and row.get("task") != "all_pauli"
    ]
    random = group_rows(random_rows, ("n_qubits", "layers", "t_density", "repeat"))
    rows.append(
        {
            "source": "random_selected_tasks",
            **regret_summary(
                {key: [as_float(row, "k90") for row in group] for key, group in random.items()}
            ),
            "decision_object": "same-circuit selected random-grid task families",
        }
    )

    floquet_rows = [
        row
        for row in read_csv("r41_kicked_ising.csv")
        if row.get("task") != "all_pauli"
    ]
    floquet = group_rows(floquet_rows, ("model", "periods"))
    rows.append(
        {
            "source": "floquet_selected_tasks",
            **regret_summary(
                {key: [as_float(row, "k90") for row in group] for key, group in floquet.items()}
            ),
            "decision_object": "same-dynamics selected Floquet task families",
        }
    )

    mqt_rows = [
        row
        for row in read_csv("r41_benchmark_validation.csv")
        if row.get("status") == "ok" and row.get("task") != "all_pauli"
    ]
    mqt = group_rows(mqt_rows, ("benchmark", "size"))
    rows.append(
        {
            "source": "MQT_selected_tasks",
            **regret_summary(
                {key: [as_float(row, "k90") for row in group] for key, group in mqt.items()}
            ),
            "decision_object": "public circuits with selected task family fixed after the circuit",
        }
    )

    fixed_rows = [
        row
        for row in read_csv("r41_fixed_error_budget.csv")
        if row.get("task") != "all_pauli" and row.get("status") == "ok"
    ]
    fixed_groups = group_rows(fixed_rows, ("dataset", "circuit_id"))
    rows.append(
        {
            "source": "fixed_error_selected_tasks",
            **regret_summary(
                {key: [row_metric(row) for row in group] for key, group in fixed_groups.items()}
            ),
            "decision_object": "fixed-error retained terms among selected local tasks",
        }
    )

    return rows


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("task") != "all_pauli" and row.get("status", "ok") == "ok"]


def row_metric(row: dict[str, str]) -> float:
    for key in ("k90", "avg_retained_terms"):
        value = as_float(row, key)
        if finite_positive(value):
            return value
    return math.nan


def selected_over_all_ratios(rows: list[dict[str, str]]) -> list[float]:
    ratios: list[float] = []
    grouped = group_rows(rows, ("circuit_id", "family", "frame_depth", "n_qubits", "benchmark", "size"))
    for group in grouped.values():
        all_values = [row_metric(row) for row in group if row.get("task") == "all_pauli"]
        all_values = [value for value in all_values if finite_positive(value)]
        if not all_values:
            continue
        all_value = all_values[0]
        for row in selected_rows(group):
            direct_ratio = as_float(row, "k90_over_all_pauli")
            if finite_positive(direct_ratio):
                ratios.append(direct_ratio)
                continue
            value = row_metric(row)
            if finite_positive(value):
                ratios.append(value / all_value)
    return ratios


def terminal_term_value(row: dict[str, str]) -> float:
    value = as_float(row, "avg_terminal_terms")
    if finite_positive(value):
        return value
    return as_float(row, "avg_retained_terms")


def summarize_cost_block(source: str, rows: list[dict[str, str]]) -> dict[str, object]:
    clean = selected_rows(rows)
    labels = [as_float(row, "labels_used") for row in clean if finite_positive(as_float(row, "labels_used"))]
    terms = [
        terminal_term_value(row)
        for row in clean
        if finite_positive(terminal_term_value(row))
    ]
    ratios = selected_over_all_ratios(rows)
    truncated = sum(as_int(row, "capped_labels") for row in clean)
    max_labels = max(labels) if labels else math.nan
    max_terms = max(terms) if terms else math.nan
    return {
        "source": source,
        "task_rows": len(clean),
        "median_selected_input_columns": fmt(median(labels)) if labels else "",
        "max_selected_input_columns": fmt(max_labels),
        "median_terminal_terms_per_column": fmt(median(terms)) if terms else "",
        "max_terminal_terms_per_column": fmt(max_terms),
        "median_selected_over_all_task": fmt(median(ratios)) if ratios else "",
        "truncated_selected_rows": truncated,
        "validation_result": "task input columns enumerated",
    }


def make_task_column_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    large_rows = read_csv("r41_large_n_separation.csv")
    rows.append(summarize_cost_block("fixed_depth_family_all_n", large_rows))
    rows.append(
        summarize_cost_block(
            "fixed_depth_family_n96",
            [row for row in large_rows if row.get("n_qubits") == "96"],
        )
    )

    fixed_rows = read_csv("r41_fixed_error_budget.csv")
    rows.append(summarize_cost_block("fixed_error_random", [row for row in fixed_rows if row.get("dataset") == "random"]))
    rows.append(summarize_cost_block("fixed_error_floquet", [row for row in fixed_rows if row.get("dataset") == "floquet"]))
    rows.append(summarize_cost_block("fixed_error_MQT_core", [row for row in fixed_rows if row.get("dataset") == "mqt_core"]))

    mqt_rows = read_csv("r41_benchmark_validation.csv")
    rows.append(summarize_cost_block("MQT_Bench_validation", mqt_rows))

    return rows


def main() -> None:
    blind_fields = [
        "source",
        "groups",
        "median_same_channel_ratio",
        "max_same_channel_ratio",
        "median_optimal_blind_regret",
        "max_optimal_blind_regret",
        "decision_object",
    ]
    cost_fields = [
        "source",
        "task_rows",
        "median_selected_input_columns",
        "max_selected_input_columns",
        "median_terminal_terms_per_column",
        "max_terminal_terms_per_column",
        "median_selected_over_all_task",
        "truncated_selected_rows",
        "validation_result",
    ]
    write_csv("r41_observable_blind_validation.csv", make_blind_validation(), blind_fields)
    write_csv("r41_selected_only_blind_validation.csv", make_selected_blind_validation(), blind_fields)
    write_csv("r41_task_column_validation.csv", make_task_column_validation(), cost_fields)


if __name__ == "__main__":
    main()
