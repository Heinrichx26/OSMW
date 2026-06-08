from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summary = json.loads((DATA_DIR / "r41_estimation_workload_benchmark.json").read_text())
        plot_benchmark(summary)
        plot_exact_dressed_trichotomy()
        return

    summary = run_quick_validation()
    print(json.dumps({"validation": summary}, indent=2))
    if args.quick:
        if summary["passed"] is False:
            raise SystemExit(1)
        return

    (DATA_DIR / "r41_estimation_workload_benchmark.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_csv(DATA_DIR / "r41_estimation_workload_benchmark.csv", flatten_summary(summary))
    plot_benchmark(summary)
    plot_exact_dressed_trichotomy()


def run_quick_validation() -> dict[str, object]:
    summary = build_summary()
    checks = [
        {
            "case": "conditioned_predictor_exceeds_global_predictor",
            "observed": summary["retained_support"]["task_p2_r2"],
            "reference": summary["retained_support"]["global_r2"],
            "passed": summary["retained_support"]["task_p2_r2"]
            > summary["retained_support"]["global_r2"],
        },
        {
            "case": "collision_estimator_collapse",
            "observed": summary["collision_estimator"]["median_rmse_ratio"],
            "threshold": 1.2,
            "passed": summary["collision_estimator"]["median_rmse_ratio"] < 1.2,
        },
        {
            "case": "one_dimensional_extreme_gap",
            "observed": summary["local_endpoint_extremes"]["one_dimensional_selected_all"],
            "threshold": 1e-50,
            "passed": summary["local_endpoint_extremes"]["one_dimensional_selected_all"] < 1e-50,
        },
        {
            "case": "two_dimensional_extreme_gap",
            "observed": summary["local_endpoint_extremes"]["two_dimensional_selected_all"],
            "threshold": 1e-15,
            "passed": summary["local_endpoint_extremes"]["two_dimensional_selected_all"] < 1e-15,
        },
    ]
    summary["checks"] = checks
    summary["passed"] = all(bool(check["passed"]) for check in checks)
    return summary


def build_summary() -> dict[str, object]:
    retained = retained_support_summary()
    collision = collision_estimator_summary()
    truncation = response_truncation_summary()
    endpoints = local_endpoint_extreme_summary()
    return {
        "case": "fixed_error_observable_estimation_workload_benchmark",
        "retained_support": retained,
        "collision_estimator": collision,
        "response_truncation": truncation,
        "local_endpoint_extremes": endpoints,
    }


def retained_support_summary() -> dict[str, object]:
    rows = read_csv(DATA_DIR / "r41_predictor_ablation.csv")
    by_predictor = {str(row["predictor"]): row for row in rows if row["target"] == "log_k90"}
    return {
        "pairs": int(float(by_predictor["task_p2"]["pairs"])),
        "global_r2": float(by_predictor["global_operator_sre"]["coefficient_of_determination"]),
        "t_count_r2": float(by_predictor["t_count"]["coefficient_of_determination"]),
        "active_t_count_r2": float(by_predictor["active_t_count"]["coefficient_of_determination"]),
        "task_p2_r2": float(by_predictor["task_p2"]["coefficient_of_determination"]),
        "task_p2_pearson": float(by_predictor["task_p2"]["pearson_correlation"]),
    }


def collision_estimator_summary() -> dict[str, object]:
    rows = read_csv(DATA_DIR / "r41_estimator_collapse.csv")
    ratios = [
        float(row["relative_rmse"]) / max(float(row["theory_relative_rmse"]), 1e-300)
        for row in rows
        if int(float(row["capped_labels"])) == 0 and float(row["effective_samples"]) >= 1.0
    ]
    local_rows = read_csv(DATA_DIR / "r41_stress_estimators.csv")
    local_ratios = [
        float(row["rmse_over_theory"])
        for row in local_rows
        if int(float(row["capped_labels"])) == 0 and float(row["effective_samples"]) >= 1.0
    ]
    all_ratios = ratios + local_ratios
    return {
        "rows": len(all_ratios),
        "median_rmse_ratio": float(np.median(all_ratios)),
        "max_rmse_ratio": float(np.max(all_ratios)),
        "local_window_max_rmse_ratio": float(np.max(local_ratios)),
    }


def response_truncation_summary() -> dict[str, object]:
    rows = read_csv(DATA_DIR / "r41_observable_truncation_error.csv")
    useful = [
        row
        for row in rows
        if int(float(row["capped_labels"])) == 0
        and str(row["state_response"]) == "tilted_product"
        and float(row["target_mass"]) >= 0.9
    ]
    relative = [float(row["relative_rms_error"]) for row in useful]
    terms = [float(row["avg_retained_terms"]) for row in useful]
    return {
        "rows": len(useful),
        "median_relative_rms_error": float(np.median(relative)),
        "max_relative_rms_error": float(np.max(relative)),
        "median_retained_terms": float(np.median(terms)),
    }


def local_endpoint_extreme_summary() -> dict[str, object]:
    one_d = json.loads((DATA_DIR / "r41_stress_workloads.json").read_text())
    two_d = json.loads((DATA_DIR / "r41_2d_stress_workloads.json").read_text())
    return {
        "one_dimensional_selected_all": float(
            one_d["summary"]["max_selected_over_all_at_largest_n"]
        ),
        "one_dimensional_k90_per_window": float(
            one_d["summary"]["width5_k90_per_window"]["window_xy_ensemble"]
        ),
        "two_dimensional_selected_all": float(
            two_d["summary"]["plaquette_xy_selected_over_all"]
        ),
        "two_dimensional_k90_per_plaquette": float(
            two_d["summary"]["plaquette_xy_k90_per_plaquette"]
        ),
    }


def plot_benchmark(summary: dict[str, object]) -> None:
    retained = summary["retained_support"]
    collision = summary["collision_estimator"]
    truncation = summary["response_truncation"]
    endpoints = summary["local_endpoint_extremes"]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.15), dpi=300)
    predictors = ["global_r2", "t_count_r2", "active_t_count_r2", "task_p2_r2"]
    labels = [r"$M_2^{\rm op}$", r"$T$", r"$T_{\rm cone}$", r"$\mathcal{P}_2$"]
    values = [float(retained[key]) for key in predictors]
    axes[0].bar(range(len(values)), values, color=["#b8b8b8", "#9a9a9a", "#707070", "#111111"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(range(len(values)), labels, rotation=0)
    axes[0].set_ylabel(r"$R^2$ for $\log K_{0.9}$")
    axes[0].set_title("(a) Retained support", fontsize=8.0, pad=2)

    axes[1].bar(
        [0, 1, 2],
        [
            float(collision["median_rmse_ratio"]),
            float(collision["max_rmse_ratio"]),
            float(collision["local_window_max_rmse_ratio"]),
        ],
        color=["#111111", "#707070", "#9a9a9a"],
    )
    axes[1].set_xticks([0, 1, 2], ["median", "all", "local"], rotation=0)
    axes[1].set_ylabel("RMSE / prediction")
    axes[1].set_title("(b) Collision law", fontsize=8.0, pad=2)

    axes[2].bar(
        [0, 1],
        [
            float(endpoints["one_dimensional_selected_all"]),
            float(endpoints["two_dimensional_selected_all"]),
        ],
        color=["#111111", "#707070"],
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks([0, 1], ["1D", "2D"])
    axes[2].set_ylabel(r"$K_{0.9}/K_{0.9}({\rm all})$")
    axes[2].set_title("(c) Local endpoints", fontsize=8.0, pad=2)
    for ax in axes:
        ax.tick_params(axis="both", length=3)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.25, w_pad=0.4)
    fig.savefig(RESULTS_DIR / "fig_estimation_workload_benchmark.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig_estimation_workload_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def plot_exact_dressed_trichotomy(eta: float = 0.9) -> None:
    n_values = np.arange(1, 97)
    local_support = np.ones_like(n_values, dtype=float)
    string_support = np.ceil(eta * np.power(2.0, n_values))
    all_pauli_support = np.ceil(eta * np.power(4.0, n_values))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(3.28, 2.15), dpi=300)
    ax.plot(n_values, np.log10(local_support), color="#111111", linewidth=1.6, label="local density")
    ax.plot(n_values, np.log10(string_support), color="#666666", linewidth=1.4, label="Pauli string")
    ax.plot(n_values, np.log10(all_pauli_support), color="#a0a0a0", linewidth=1.4, label="all Pauli")
    ax.set_xlabel("qubits")
    ax.set_ylabel(r"$\log_{10} K_{0.9}$")
    ax.set_title("Exact same-channel phases", fontsize=8.0, pad=2)
    ax.set_xlim(1, 96)
    ax.set_ylim(-1.0, 59.5)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.35)
    fig.savefig(RESULTS_DIR / "fig1_exact_dressed_trichotomy.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig1_exact_dressed_trichotomy.png", bbox_inches="tight")
    plt.close(fig)


def flatten_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, payload in summary.items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                rows.append({"group": group, "metric": key, "value": value})
    return rows


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: convert_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def convert_value(value: str) -> object:
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
