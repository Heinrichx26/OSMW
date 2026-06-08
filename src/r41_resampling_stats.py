from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _read_global_task_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = DATA_DIR / "r41_global_task_truncation.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "t_density": float(row["t_density"]),
                    "global_p2_estimate": float(row["global_p2_estimate"]),
                    "task_p2": float(row["task_p2"]),
                    "k90": float(row["k90"]),
                }
            )
    return rows


def _read_fixed_error_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = DATA_DIR / "r41_fixed_error_budget_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "median_ratio": float(row["median_ratio"]),
                    "q25_ratio": float(row["q25_ratio"]),
                    "q75_ratio": float(row["q75_ratio"]),
                }
            )
    return rows


def _r2(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if len(x_values) < 2:
        return float("nan")
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    prediction = slope * x_values + intercept
    ss_res = float(np.sum((y_values - prediction) ** 2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def _bootstrap_ci(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_rows = len(x_values)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, n_rows, size=n_rows)
        values[draw] = _r2(x_values[indices], y_values[indices])
    return (
        _r2(x_values, y_values),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _fit_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    subsets = {
        "all": rows,
        "n12_density_0.20": [
            row for row in rows if row["n_qubits"] == 12 and abs(row["t_density"] - 0.20) < 1e-12
        ],
    }
    out: list[dict[str, object]] = []
    seed_base = 3112026
    for subset_name, subset_rows in subsets.items():
        y_values = np.log(np.array([row["k90"] for row in subset_rows], dtype=float))
        for offset, (predictor, field) in enumerate(
            [
                ("global_operator_sre", "global_p2_estimate"),
                ("task_entropy", "task_p2"),
            ]
        ):
            x_values = np.array([row[field] for row in subset_rows], dtype=float)
            r2, low, high = _bootstrap_ci(
                x_values,
                y_values,
                seed=seed_base + 17 * offset + len(subset_rows),
            )
            out.append(
                {
                    "subset": subset_name,
                    "predictor": predictor,
                    "pairs": len(subset_rows),
                    "r2": r2,
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                }
            )
    return out


def _hierarchy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    task_order = ["local_z", "local_transverse", "hamiltonian_terms"]
    labels = {"local_z": "Z", "local_transverse": "X/Y", "hamiltonian_terms": "H"}
    datasets = sorted({row["dataset"] for row in rows})
    lookup = {(row["dataset"], row["task"]): row for row in rows}
    out: list[dict[str, object]] = []
    for dataset in datasets:
        medians = [lookup[(dataset, task)]["median_ratio"] for task in task_order]
        out.append(
            {
                "dataset": dataset,
                "ordering": "<".join(labels[task] for task in task_order),
                "median_ratios": medians,
                "monotone": bool(medians[0] < medians[1] < medians[2]),
            }
        )
    return out


def main() -> None:
    fit_rows = _fit_rows(_read_global_task_rows())
    hierarchy_rows = _hierarchy(_read_fixed_error_rows())

    csv_path = DATA_DIR / "r41_resampling_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subset", "predictor", "pairs", "r2", "bootstrap_low", "bootstrap_high"],
        )
        writer.writeheader()
        writer.writerows(fit_rows)

    json_path = DATA_DIR / "r41_resampling_stats.json"
    json_path.write_text(
        json.dumps({"predictor_bootstrap": fit_rows, "fixed_error_hierarchy": hierarchy_rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
