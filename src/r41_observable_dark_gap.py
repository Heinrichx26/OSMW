from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


TASKS = ("local_z", "local_transverse", "hamiltonian_terms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    source = DATA_DIR / "r41_large_n_separation.csv"
    rows = read_rows(source)
    gap_rows, summary = compute_dark_gap(rows)
    validation = validate(gap_rows, summary)
    payload = {"validation": validation, "summary": summary}
    print(json.dumps(payload, indent=2))
    if args.quick:
        if not validation["passed"]:
            raise SystemExit(1)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "r41_observable_dark_gap.csv", gap_rows)
    (DATA_DIR / "r41_observable_dark_gap.json").write_text(
        json.dumps({**payload, "rows": gap_rows}, indent=2),
        encoding="utf-8",
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [coerce(row) for row in csv.DictReader(handle)]


def coerce(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = dict(row)
    for key in [
        "n_qubits",
        "operator_magic",
        "task_p2",
        "exp_task_p2",
        "k90",
        "log10_k90",
        "terminal_support",
        "labels_used",
        "labels_requested",
        "capped_labels",
        "avg_terminal_terms",
        "log10_k90_over_all_pauli",
        "k90_over_all_pauli",
    ]:
        value = row.get(key, "")
        if value in {"", "nan", "NaN"}:
            out[key] = math.nan
        else:
            out[key] = float(value)
    out["n_qubits"] = int(float(row["n_qubits"]))
    return out


def compute_dark_gap(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_qubits"]), {})[str(row["task"])] = row

    gap_rows: list[dict[str, object]] = []
    for n_qubits in sorted(grouped):
        task_rows = grouped[n_qubits]
        all_row = task_rows["all_pauli"]
        all_log10 = float(all_row["log10_k90"])
        for task in TASKS:
            row = task_rows[task]
            task_log10 = float(row["log10_k90"])
            dark_gap_log10 = all_log10 - task_log10
            labels_used = float(row["labels_used"])
            per_term_k90 = float(row["k90"]) / labels_used
            gap_rows.append(
                {
                    "n_qubits": n_qubits,
                    "task": task,
                    "log10_k90_all": all_log10,
                    "log10_k90_task": task_log10,
                    "dark_gap_log10": dark_gap_log10,
                    "dark_gap_density": dark_gap_log10 / n_qubits,
                    "visible_density": task_log10 / n_qubits,
                    "dark_fraction_of_all": dark_gap_log10 / all_log10,
                    "k90_per_term": per_term_k90,
                }
            )

    largest_n = max(grouped)
    summary_tasks: dict[str, object] = {}
    for task in TASKS:
        subset = [row for row in gap_rows if row["task"] == task]
        fit_gap = linear_fit(
            [float(row["n_qubits"]) for row in subset],
            [float(row["dark_gap_log10"]) for row in subset],
        )
        fit_visible = linear_fit(
            [float(row["n_qubits"]) for row in subset],
            [float(row["log10_k90_task"]) for row in subset],
        )
        last = next(row for row in subset if int(row["n_qubits"]) == largest_n)
        summary_tasks[task] = {
            "dark_gap_slope_log10_per_qubit": fit_gap["slope"],
            "dark_gap_fit_r2": fit_gap["r2"],
            "visible_slope_log10_per_qubit": fit_visible["slope"],
            "visible_fit_r2": fit_visible["r2"],
            "dark_gap_density_at_largest_n": last["dark_gap_density"],
            "visible_density_at_largest_n": last["visible_density"],
            "dark_fraction_at_largest_n": last["dark_fraction_of_all"],
            "k90_per_term_at_largest_n": last["k90_per_term"],
        }

    return gap_rows, {
        "largest_n": largest_n,
        "all_pauli_density_log10": math.log10(4.0),
        "tasks": summary_tasks,
    }


def linear_fit(x_values: list[float], y_values: list[float]) -> dict[str, float]:
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x in x_values]
    ss_res = sum((y - pred) ** 2 for y, pred in zip(y_values, predictions))
    ss_tot = sum((y - mean_y) ** 2 for y in y_values)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def validate(gap_rows: list[dict[str, object]], summary: dict[str, object]) -> dict[str, object]:
    largest_n = int(summary["largest_n"])
    checks: list[dict[str, object]] = [
        {
            "case": "largest_n_at_least_96",
            "observed": largest_n,
            "passed": largest_n >= 96,
        }
    ]
    tasks = summary["tasks"]
    for task in TASKS:
        item = tasks[task]
        checks.extend(
            [
                {
                    "case": f"{task}_dark_gap_density_large",
                    "observed": item["dark_gap_density_at_largest_n"],
                    "passed": item["dark_gap_density_at_largest_n"] > 0.55,
                },
                {
                    "case": f"{task}_visible_density_small",
                    "observed": item["visible_density_at_largest_n"],
                    "passed": item["visible_density_at_largest_n"] < 0.05,
                },
                {
                    "case": f"{task}_dark_gap_linear",
                    "observed": item["dark_gap_fit_r2"],
                    "passed": item["dark_gap_fit_r2"] > 0.999,
                },
            ]
        )
    return {"case": "observable_dark_gap_from_large_n_data", "checks": checks, "passed": all(bool(check["passed"]) for check in checks)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
