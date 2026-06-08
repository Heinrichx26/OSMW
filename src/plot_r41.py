from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")


def _panel_title(index: int, title: str) -> str:
    return f"{PANEL_LABELS[index]}{title}"


def _legend_items(*axes):
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_") or label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    return handles, labels


def _finish_with_legend(
    fig,
    axes,
    *,
    ncol: int,
    bottom: float,
    pad: float = 0.25,
    w_pad: float = 0.35,
    fontsize: float | None = None,
    legend_y: float = 0.0,
) -> None:
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=pad, w_pad=w_pad)
    handles, labels = _legend_items(*axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            frameon=False,
            ncol=ncol,
            fontsize=fontsize,
            handlelength=1.4,
            handletextpad=0.45,
            columnspacing=0.9,
            labelspacing=0.25,
            borderaxespad=0.0,
        )


def main() -> None:
    scatter_rows = _read_scatter(RESULTS_DIR / "r41_scatter_path_vs_magic.csv")
    extremes = json.loads((RESULTS_DIR / "r41_extreme_examples.json").read_text(encoding="utf-8"))
    growth = json.loads((RESULTS_DIR / "r41_growth_fits.json").read_text(encoding="utf-8"))

    _plot_scatter(scatter_rows, extremes)
    _plot_growth(growth)
    _plot_exact_same_channel_separation()
    global_task_rows: list[dict[str, object]] = []
    kicked_rows: list[dict[str, object]] = []
    scaling_path = RESULTS_DIR / "r41_pauli_scaling.csv"
    if scaling_path.exists():
        _plot_pauli_scaling(_read_scaling(scaling_path))
    large_n_path = DATA_DIR / "r41_large_n_separation.csv"
    if large_n_path.exists():
        _plot_large_n_separation(_read_large_n_rows(large_n_path))
    task_path = RESULTS_DIR / "r41_task_resolved.csv"
    if task_path.exists():
        _plot_task_resolved(_read_task_rows(task_path))
    global_task_path = DATA_DIR / "r41_global_task_truncation.csv"
    if global_task_path.exists():
        global_task_rows = _read_global_task_rows(global_task_path)
        _plot_global_task_truncation(global_task_rows)
        _plot_predictor_benchmark(global_task_rows)
    observable_path = DATA_DIR / "r41_observable_task_scaling.csv"
    if observable_path.exists():
        _plot_observable_tasks(_read_observable_task_rows(observable_path))
    kicked_path = DATA_DIR / "r41_kicked_ising.csv"
    if kicked_path.exists():
        kicked_rows = _read_kicked_rows(kicked_path)
        _plot_kicked_ising(kicked_rows)
    if global_task_rows and kicked_rows:
        _plot_same_channel_spread(global_task_rows, kicked_rows)
    collapse_path = DATA_DIR / "r41_estimator_collapse.csv"
    if collapse_path.exists():
        truncation_path = DATA_DIR / "r41_truncation_profile.csv"
        truncation_rows = _read_truncation_rows(truncation_path) if truncation_path.exists() else []
        observable_path = DATA_DIR / "r41_observable_truncation_error.csv"
        observable_rows = (
            _read_observable_truncation_rows(observable_path)
            if observable_path.exists()
            else []
        )
        _plot_estimator_collapse(
            _read_collapse_rows(collapse_path),
            truncation_rows,
            observable_rows,
        )
    benchmark_path = DATA_DIR / "r41_benchmark_validation.csv"
    if benchmark_path.exists():
        _plot_mqt_benchmark(_read_benchmark_rows(benchmark_path))


def _read_scatter(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "depth": int(row["depth"]),
                    "median_c2": float(row["median_c2"]),
                    "operator_magic": float(row["operator_magic"]),
                }
            )
    return rows


def _read_scaling(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "expected_t_gates": float(row["expected_t_gates"]),
                    "p2_mean": float(row["p2_mean"]),
                    "p2_std_over_circuits": float(row["p2_std_over_circuits"]),
                }
            )
    return rows


def _read_task_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "task_p2": float(row["task_p2"]),
                    "operator_magic": float(row["operator_magic"]),
                }
            )
    return rows


def _read_observable_task_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "expected_t_gates": float(row["expected_t_gates"]),
                    "task_p2_mean": float(row["task_p2_mean"]),
                    "task_p2_sem_over_circuits": float(row["task_p2_sem_over_circuits"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _read_global_task_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "layers": int(row["layers"]),
                    "t_density": float(row["t_density"]),
                    "expected_t_gates": float(row["expected_t_gates"]),
                    "repeat": int(row["repeat"]),
                    "global_p2_estimate": float(row["global_p2_estimate"]),
                    "global_k90_estimate": int(row["global_k90_estimate"]),
                    "task_p2": float(row["task_p2"]),
                    "exp_task_p2": float(row["exp_task_p2"]),
                    "k90": int(row["k90"]),
                    "k90_lower_bound": float(row["k90_lower_bound"]),
                    "k90_over_exp_p2": float(row["k90_over_exp_p2"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _read_collapse_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "task_p2": float(row["task_p2"]),
                    "collision_probability": float(row["collision_probability"]),
                    "n_qubits": int(row.get("n_qubits", 0) or 0),
                    "samples": int(row["samples"]),
                    "effective_samples": float(row["effective_samples"]),
                    "relative_rmse": float(row["relative_rmse"]),
                    "theory_relative_rmse": float(row["theory_relative_rmse"]),
                    "capped_labels": int(row.get("capped_labels", 0) or 0),
                }
            )
    return rows


def _read_kicked_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "periods": int(row["periods"]),
                    "task_p2": float(row["task_p2"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _read_truncation_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "task_p2": float(row["task_p2"]),
                    "exp_task_p2": float(row["exp_task_p2"]),
                    "marginal_effective_support": float(row["marginal_effective_support"]),
                    "terminal_support": int(row["terminal_support"]),
                    "k50": int(row["k50"]),
                    "k90": int(row["k90"]),
                    "k95": int(row["k95"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _read_observable_truncation_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "n_qubits": int(row["n_qubits"]),
                    "target_mass": float(row["target_mass"]),
                    "mean_retained_mass": float(row["mean_retained_mass"]),
                    "avg_retained_terms": float(row["avg_retained_terms"]),
                    "rms_error": float(row["rms_error"]),
                    "relative_rms_error": float(row["relative_rms_error"]),
                    "exact_rms": float(row["exact_rms"]),
                    "task_p2": float(row["task_p2"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _read_benchmark_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["status"] != "ok":
                continue
            rows.append(
                {
                    **row,
                    "size": int(row["size"]),
                    "n_qubits": int(row["n_qubits"]),
                    "gates": int(row["gates"]),
                    "t_count": int(row["t_count"]),
                    "task_p2": float(row["task_p2"]),
                    "k90": int(row["k90"]),
                    "global_p2_estimate": float(row["global_p2_estimate"]),
                    "global_k90_estimate": int(float(row["global_k90_estimate"])),
                }
            )
    return rows


def _read_fixed_error_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "circuits": int(row["circuits"]),
                    "median_ratio": float(row["median_ratio"]),
                    "q25_ratio": float(row["q25_ratio"]),
                    "q75_ratio": float(row["q75_ratio"]),
                }
            )
    return rows


def _read_predictor_ablation(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "pairs": int(row["pairs"]),
                    "slope": float(row["slope"]),
                    "intercept": float(row["intercept"]),
                    "coefficient_of_determination": float(
                        row["coefficient_of_determination"]
                    ),
                    "pearson_correlation": float(row["pearson_correlation"]),
                }
            )
    return rows


def _read_large_n_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    **row,
                    "frame_depth": int(row["frame_depth"]),
                    "n_qubits": int(row["n_qubits"]),
                    "operator_magic": float(row["operator_magic"]),
                    "task_p2": float(row["task_p2"]),
                    "exp_task_p2": float(row["exp_task_p2"]),
                    "k90": float(row["k90"]),
                    "log10_k90": float(row["log10_k90"]),
                    "log10_k90_over_all_pauli": float(row["log10_k90_over_all_pauli"]),
                    "k90_over_all_pauli": float(row["k90_over_all_pauli"]),
                    "labels_used": float(row["labels_used"]),
                    "avg_terminal_terms": float(row["avg_terminal_terms"]),
                    "capped_labels": int(row["capped_labels"]),
                }
            )
    return rows


def _plot_scatter(rows: list[dict[str, object]], extremes: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=300)
    samples = [row for row in rows if row["source"] == "sample"]
    constructed = [row for row in rows if row["source"] == "constructed"]

    ax.scatter(
        [row["median_c2"] for row in samples],
        [row["operator_magic"] for row in samples],
        s=10,
        c="#8a8a8a",
        alpha=0.45,
        linewidths=0,
        label="sampled circuits",
    )
    ax.scatter(
        [row["median_c2"] for row in constructed],
        [row["operator_magic"] for row in constructed],
        s=18,
        c="#222222",
        alpha=0.8,
        linewidths=0,
        label="constructed families",
    )

    for key, marker in [("high_path_low_magic", "s"), ("low_path_high_magic", "D")]:
        row = extremes[key]
        ax.scatter(
            [row["median_c2"]],
            [row["operator_magic"]],
            s=44,
            facecolors="none",
            edgecolors="#000000",
            linewidths=1.0,
        marker=marker,
        label=key.replace("_", " "),
        )

    ax.set_xlabel(r"path participation $C_2$")
    ax.set_ylabel(r"operator magic $M_2^{\mathrm{op}}$")
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="center right", handletextpad=0.4, borderpad=0.2)
    ax.grid(True, linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.25)
    _save(fig, "fig2_path_vs_magic")


def _plot_growth(growth: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=300)
    styles = {
        "simple_clifford": {"label": "simple Clifford", "marker": "o", "linestyle": "-"},
        "brickwork_clifford_n2": {"label": "brickwork Clifford", "marker": "s", "linestyle": "--"},
    }
    for key, style in styles.items():
        item = growth[key]
        ax.plot(
            item["depths"],
            item["median_c2"],
            color="#111111" if key == "brickwork_clifford_n2" else "#777777",
            marker=style["marker"],
            markersize=3.2,
            linewidth=0.9,
            linestyle=style["linestyle"],
            label=f"{style['label']}, $R^2={item['fit']['r2']:.2f}$",
        )
    ax.set_xlabel(r"family depth $L$")
    ax.set_ylabel(r"median $C_2$")
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    ax.grid(True, linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.25)
    _save(fig, "fig3_clifford_growth")


def _plot_pauli_scaling(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=300)
    styles = {
        8: {"marker": "o", "label": "$n=8$", "color": "#111111", "linestyle": "-"},
        12: {"marker": "s", "label": "$n=12$", "color": "#666666", "linestyle": "--"},
        16: {"marker": "^", "label": "$n=16$", "color": "#999999", "linestyle": "-."},
    }
    for n_qubits, style in styles.items():
        subset = sorted(
            [row for row in rows if row["n_qubits"] == n_qubits],
            key=lambda row: row["expected_t_gates"],
        )
        ax.errorbar(
            [row["expected_t_gates"] for row in subset],
            [row["p2_mean"] for row in subset],
            yerr=[row["p2_std_over_circuits"] for row in subset],
            marker=style["marker"],
            markersize=3.2,
            linewidth=0.8,
            linestyle=style["linestyle"],
            color=style["color"],
            capsize=1.8,
            label=style["label"],
        )
        capped = [row for row in subset if int(row.get("capped_samples", 0)) > 0]
        if capped:
            ax.plot(
                [row["expected_t_gates"] for row in capped],
                [row["p2_mean"] for row in capped],
                linestyle="none",
                marker=style["marker"],
                markersize=5.0,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
                zorder=5,
            )
    xmax = max(row["expected_t_gates"] for row in rows)
    xs = [0.0, xmax]
    ax.plot(xs, [x * 0.28768207245178085 for x in xs], color="#111111", linewidth=0.8, linestyle="--", label=r"$N_T\log(4/3)$")
    ax.set_xlabel(r"expected $T$ gates $N_T$")
    ax.set_ylabel(r"Pauli-path $\mathcal{P}_2$")
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    ax.grid(True, linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.25)
    _save(fig, "fig4_pauli_scaling")


def _plot_task_resolved(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=300)
    task_styles = {
        "z_sector": {"label": "$Z$ sector", "marker": "o", "linestyle": "-", "color": "#111111"},
        "all_paulis": {"label": "all Pauli tasks", "marker": "s", "linestyle": "--", "color": "#555555"},
        "xy_all_sector": {"label": "full $X/Y$ sector", "marker": "^", "linestyle": "-.", "color": "#888888"},
        "xy_one_site_sector": {"label": "one-site $X/Y$", "marker": "D", "linestyle": ":", "color": "#222222"},
    }
    xs = list(range(1, 7))
    analytic = {
        "z_sector": [0.0 for _ in xs],
        "all_paulis": [x * 0.28768207245178085 for x in xs],
        "xy_all_sector": [x * 0.6931471805599453 for x in xs],
        "xy_one_site_sector": [0.6931471805599453 for _ in xs],
    }
    for task, style in task_styles.items():
        ax.plot(
            xs,
            analytic[task],
            marker=style["marker"],
            markersize=3.2,
            linewidth=0.9,
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel(r"qubits $n$")
    ax.set_ylabel(r"task-resolved $\mathcal{P}_2$")
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    ax.grid(True, linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.25)
    _save(fig, "fig5_task_resolved")
    _save(fig, "fig2_task_separation")


def _plot_exact_same_channel_separation() -> None:
    eta = 0.9
    n_values = np.arange(1, 97)
    fig, ax = plt.subplots(figsize=(3.6, 2.05), dpi=300)
    ax.plot(
        n_values,
        np.zeros_like(n_values, dtype=float),
        color="#111111",
        linewidth=0.9,
        linestyle="-",
        label=r"$K_\eta(Z_1)=1$",
    )
    ax.plot(
        n_values,
        np.log10(np.ceil(eta * (2.0**n_values))),
        color="#555555",
        linewidth=0.9,
        linestyle="--",
        label=r"$K_\eta(X^{\otimes n})$",
    )
    ax.plot(
        n_values,
        n_values * math.log10(4.0 / 3.0),
        color="#9a9a9a",
        linewidth=0.9,
        linestyle=":",
        label=r"$e^{M_2^{\rm op}}$",
    )
    ax.set_xlabel("qubits")
    ax.set_ylabel(r"$\log_{10}$ scale")
    ax.set_title(_panel_title(1, r"Exact $T^{\otimes n}$ separation"), fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, linewidth=0.3, alpha=0.22)
    _finish_with_legend(
        fig,
        [ax],
        ncol=3,
        bottom=0.20,
        pad=0.18,
        w_pad=0.0,
        fontsize=5.2,
        legend_y=0.026,
    )
    _save(fig, "fig1_exact_same_channel")


def _plot_large_n_separation(rows: list[dict[str, object]]) -> None:
    task_order = ["local_z", "local_transverse", "hamiltonian_terms", "all_pauli"]
    labels = {
        "local_z": r"local $Z$",
        "local_transverse": r"local $X/Y$",
        "hamiltonian_terms": r"Hamiltonian terms",
        "all_pauli": r"all Pauli",
    }
    styles = {
        "local_z": ("o", "#111111", "-"),
        "local_transverse": ("s", "#4d4d4d", "-"),
        "hamiltonian_terms": ("D", "#8a8a8a", "-"),
        "all_pauli": ("^", "#111111", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.14), dpi=300)
    for task in task_order:
        marker, color, linestyle = styles[task]
        subset = [row for row in rows if row["task"] == task]
        axes[0].plot(
            [row["n_qubits"] for row in subset],
            [row["log10_k90"] for row in subset],
            marker=marker,
            markersize=3.0,
            linewidth=0.9,
            linestyle=linestyle,
            color=color,
            markerfacecolor="white" if task == "all_pauli" else color,
            markeredgewidth=0.6,
            label=labels[task],
        )
        if task != "all_pauli":
            axes[1].plot(
                [row["n_qubits"] for row in subset],
                [
                    -float(row["log10_k90_over_all_pauli"]) / float(row["n_qubits"])
                    for row in subset
                ],
                marker=marker,
                markersize=3.0,
                linewidth=0.9,
                linestyle="-",
                color=color,
                markerfacecolor=color,
                markeredgewidth=0.6,
                label=labels[task],
            )
        if task != "all_pauli":
            per_term = [
                row["k90"] / row["labels_used"]
                for row in subset
                if row["labels_used"] and not math.isnan(row["labels_used"])
            ]
            per_term_n = [
                row["n_qubits"]
                for row in subset
                if row["labels_used"] and not math.isnan(row["labels_used"])
            ]
            axes[2].plot(
                per_term_n,
                per_term,
                marker=marker,
                markersize=3.0,
                linewidth=0.9,
                linestyle="-",
                color=color,
                markerfacecolor=color,
                markeredgewidth=0.6,
                label=labels[task],
            )

    n_values = sorted({row["n_qubits"] for row in rows})
    axes[0].plot(
        n_values,
        [n * math.log10(4.0 / 3.0) for n in n_values],
        color="#c0c0c0",
        linewidth=0.9,
        linestyle=":",
        label=r"$e^{M_2^{\rm op}}$",
    )
    axes[0].set_xlabel("qubits")
    axes[0].set_ylabel(r"$\log_{10} K_{0.9}$")
    axes[0].set_title(_panel_title(0, "All-Pauli explosion"), fontsize=7, pad=2)

    axes[1].axhline(math.log10(4.0), color="#111111", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("qubits")
    axes[1].set_ylabel(r"$\Delta_{0.9}(O)/n$")
    axes[1].set_title(_panel_title(1, "Dictionary gap density"), fontsize=7, pad=2)
    axes[2].set_xlabel("qubits")
    axes[2].set_ylabel(r"$K_{0.9}(O)/N_{\rm terms}$")
    axes[2].set_title(_panel_title(2, "Light-cone terms"), fontsize=7, pad=2)
    for ax in axes:
        ax.tick_params(axis="both", length=3)
        ax.grid(True, linewidth=0.3, alpha=0.22)
    _finish_with_legend(
        fig,
        axes,
        ncol=5,
        bottom=0.155,
        pad=0.18,
        w_pad=0.28,
        fontsize=4.6,
        legend_y=0.026,
    )
    _save(fig, "fig1_local_magic_separation")


def _plot_kicked_ising(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.0), dpi=300, sharey=True)
    task_styles = {
        "local_z": {"label": "local $Z$", "marker": "o", "color": "#111111", "linestyle": "-"},
        "local_transverse": {
            "label": "local $X/Y$",
            "marker": "s",
            "color": "#555555",
            "linestyle": "--",
        },
        "hamiltonian_terms": {
            "label": "Hamiltonian",
            "marker": "D",
            "color": "#777777",
            "linestyle": ":",
        },
        "all_pauli": {"label": "all Pauli", "marker": "^", "color": "#000000", "linestyle": "-."},
    }
    titles = {"commuting": "Commuting kick", "scrambling": "Scrambling kick"}
    for panel_index, (ax, model) in enumerate(zip(axes, ["commuting", "scrambling"])):
        for task, style in task_styles.items():
            subset = sorted(
                [row for row in rows if row["model"] == model and row["task"] == task],
                key=lambda row: row["periods"],
            )
            ax.plot(
                [row["periods"] for row in subset],
                [row["task_p2"] for row in subset],
                color=style["color"],
                marker=style["marker"],
                markersize=3.0,
                linewidth=0.8,
                linestyle=style["linestyle"],
                label=style["label"],
            )
        ax.set_title(_panel_title(panel_index, titles[model]), fontsize=7, pad=2)
        ax.set_xlabel("Floquet periods")
        ax.tick_params(axis="both", length=3)
        ax.grid(True, linewidth=0.3, alpha=0.22)
    axes[0].set_ylabel(r"task $\mathcal{P}_2$")
    _finish_with_legend(
        fig,
        axes,
        ncol=4,
        bottom=0.105,
        pad=0.25,
        w_pad=0.35,
        legend_y=0.028,
    )
    _save(fig, "fig2_kicked_ising")


def _plot_same_channel_spread(
    global_rows: list[dict[str, object]],
    kicked_rows: list[dict[str, object]],
) -> None:
    tasks = ["local_z", "local_transverse", "hamiltonian_terms", "all_pauli"]
    labels = {
        "local_z": r"$Z$",
        "local_transverse": r"$X/Y$",
        "hamiltonian_terms": r"$H$",
        "all_pauli": "all",
    }
    styles = {
        "local_z": ("o", "#111111"),
        "local_transverse": ("s", "#444444"),
        "hamiltonian_terms": ("D", "#777777"),
        "all_pauli": ("^", "#000000"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), dpi=300)

    random_subset = [
        row
        for row in global_rows
        if row["n_qubits"] == 12
        and abs(float(row["t_density"]) - 0.20) < 1e-9
        and row["task"] in set(tasks)
    ]
    by_seed: dict[int, list[dict[str, object]]] = {}
    for row in random_subset:
        by_seed.setdefault(int(row["circuit_seed"]), []).append(row)
    ax = axes[0]
    for group in by_seed.values():
        if len(group) < 2:
            continue
        x = float(group[0]["global_p2_estimate"])
        ys = [float(row["k90"]) for row in group]
        ax.vlines(x, min(ys), max(ys), color="#d0d0d0", linewidth=0.45, zorder=0)
    for task in tasks:
        marker, color = styles[task]
        subset = [row for row in random_subset if row["task"] == task]
        ax.semilogy(
            [row["global_p2_estimate"] for row in subset],
            [row["k90"] for row in subset],
            linestyle="none",
            marker=marker,
            markersize=2.7,
            color=color,
            markerfacecolor="white" if task == "all_pauli" else color,
            markeredgewidth=0.6,
            alpha=0.78,
            label=labels[task],
        )
    ax.set_xlabel(r"global $M_2^{\mathrm{op}}$ estimate")
    ax.set_ylabel(r"$K_{0.9}$")
    ax.set_title("random circuits", fontsize=7, pad=2)
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.0)

    floquet_subset = [
        row
        for row in kicked_rows
        if row["model"] == "scrambling" and row["periods"] > 0 and row["task"] in set(tasks)
    ]
    all_by_period = {
        int(row["periods"]): row
        for row in floquet_subset
        if row["task"] == "all_pauli"
    }
    ax = axes[1]
    for period in sorted(all_by_period):
        group = [row for row in floquet_subset if int(row["periods"]) == period]
        if not group:
            continue
        x = float(all_by_period[period]["task_p2"])
        ys = [float(row["k90"]) for row in group]
        ax.vlines(x, min(ys), max(ys), color="#d0d0d0", linewidth=0.55, zorder=0)
    for task in tasks:
        marker, color = styles[task]
        subset = [row for row in floquet_subset if row["task"] == task]
        ax.semilogy(
            [float(all_by_period[int(row["periods"])]["task_p2"]) for row in subset],
            [row["k90"] for row in subset],
            linestyle="none",
            marker=marker,
            markersize=3.0,
            color=color,
            markerfacecolor="white" if task == "all_pauli" else color,
            markeredgewidth=0.6,
            alpha=0.82,
        )
    ax.set_xlabel(r"all-Pauli $\mathcal{P}_2$")
    ax.set_title("Floquet dynamics", fontsize=7, pad=2)

    ax = axes[2]
    ratio_data = []
    ratio_labels = []
    random_ratios = _same_circuit_ratios(global_rows, tasks[:-1])
    floquet_ratios = _kicked_ratios(floquet_subset, tasks[:-1])
    for source, ratios in [("Rand.", random_ratios), ("Floq.", floquet_ratios)]:
        for task in tasks[:-1]:
            ratio_data.append(ratios.get(task, []))
            ratio_labels.append(f"{source} {labels[task]}")
    ax.boxplot(
        ratio_data,
        positions=np.arange(len(ratio_data)) + 1,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.85},
        boxprops={"facecolor": "#e6e6e6", "edgecolor": "#555555", "linewidth": 0.65},
        whiskerprops={"color": "#555555", "linewidth": 0.65},
        capprops={"color": "#555555", "linewidth": 0.65},
    )
    ax.axhline(1.0, color="#111111", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_{0.9}(O)/K_{0.9}({\rm all})$")
    ax.set_xticks(np.arange(len(ratio_data)) + 1)
    ax.set_xticklabels(ratio_labels, rotation=45, ha="right")
    ax.set_title("same-channel ratio", fontsize=7, pad=2)

    for ax in axes:
        ax.tick_params(axis="both", length=3)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    fig.tight_layout(pad=0.22, w_pad=0.35)
    _save(fig, "fig2_same_channel_spread")


def _same_circuit_ratios(
    rows: list[dict[str, object]],
    tasks: list[str],
) -> dict[str, list[float]]:
    by_circuit: dict[tuple[int, int, float, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["n_qubits"]),
            int(row["layers"]),
            float(row["t_density"]),
            int(row["circuit_seed"]),
        )
        by_circuit.setdefault(key, {})[str(row["task"])] = row
    ratios = {task: [] for task in tasks}
    for group in by_circuit.values():
        if "all_pauli" not in group:
            continue
        all_k = float(group["all_pauli"]["k90"])
        if all_k <= 0:
            continue
        for task in tasks:
            if task in group:
                ratios[task].append(float(group[task]["k90"]) / all_k)
    return ratios


def _kicked_ratios(
    rows: list[dict[str, object]],
    tasks: list[str],
) -> dict[str, list[float]]:
    by_period: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_period.setdefault(int(row["periods"]), {})[str(row["task"])] = row
    ratios = {task: [] for task in tasks}
    for group in by_period.values():
        if "all_pauli" not in group:
            continue
        all_k = float(group["all_pauli"]["k90"])
        if all_k <= 0:
            continue
        for task in tasks:
            if task in group:
                ratios[task].append(float(group[task]["k90"]) / all_k)
    return ratios


def _plot_observable_tasks(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0), dpi=300, sharey=True)
    task_styles = {
        "local_z": {"label": "local $Z$", "marker": "o", "color": "#111111", "linestyle": "-"},
        "local_transverse": {
            "label": "local $X/Y$",
            "marker": "s",
            "color": "#666666",
            "linestyle": "--",
        },
        "hamiltonian_terms": {
            "label": "Hamiltonian",
            "marker": "D",
            "color": "#333333",
            "linestyle": ":",
        },
        "random_local": {
            "label": "random local",
            "marker": "v",
            "color": "#777777",
            "linestyle": (0, (3, 1, 1, 1)),
        },
        "transported_local_z": {
            "label": "transported $Z$",
            "marker": "P",
            "color": "#444444",
            "linestyle": (0, (5, 1)),
        },
        "full_transverse": {
            "label": "full $X/Y$",
            "marker": "X",
            "color": "#999999",
            "linestyle": "-.",
        },
        "all_pauli": {"label": "all Pauli", "marker": "^", "color": "#000000", "linestyle": "--"},
    }
    for panel_index, (ax, n_qubits) in enumerate(zip(axes, [8, 10, 12])):
        for task, style in task_styles.items():
            subset = sorted(
                [row for row in rows if row["n_qubits"] == n_qubits and row["task"] == task],
                key=lambda row: row["expected_t_gates"],
            )
            ax.errorbar(
                [row["expected_t_gates"] for row in subset],
                [row["task_p2_mean"] for row in subset],
                yerr=[row["task_p2_sem_over_circuits"] for row in subset],
                color=style["color"],
                marker=style["marker"],
                markersize=3.0,
                linewidth=0.8,
                linestyle=style["linestyle"],
                capsize=1.5,
                label=style["label"],
            )
        ax.set_title(_panel_title(panel_index, rf"$n={n_qubits}$"), fontsize=7, pad=2)
        ax.set_xlabel(r"$N_T$")
        ax.tick_params(axis="both", length=3)
        ax.grid(True, linewidth=0.3, alpha=0.22)
    axes[0].set_ylabel(r"task $\mathcal{P}_2$")
    _finish_with_legend(
        fig,
        axes,
        ncol=4,
        bottom=0.17,
        pad=0.25,
        w_pad=0.35,
        legend_y=0.030,
    )
    _save(fig, "fig3_observable_tasks")


def _plot_global_task_truncation(rows: list[dict[str, object]]) -> None:
    task_order = [
        "local_z",
        "local_transverse",
        "hamiltonian_terms",
        "transported_local_z",
        "all_pauli",
    ]
    labels = {
        "local_z": r"$Z_{\rm loc}$",
        "local_transverse": r"$X/Y$",
        "hamiltonian_terms": r"$H$",
        "transported_local_z": r"$\widetilde Z$",
        "all_pauli": "all",
    }
    styles = {
        "local_z": ("o", "#111111"),
        "local_transverse": ("s", "#444444"),
        "hamiltonian_terms": ("D", "#777777"),
        "transported_local_z": ("v", "#999999"),
        "all_pauli": ("^", "#000000"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), dpi=300)
    for ax, field, title, xlabel in [
        (axes[0], "global_p2_estimate", "channel average", r"global $M_2^{\mathrm{op}}$"),
        (axes[1], "task_p2", "observable selected", r"task $\mathcal{P}_2(U;O)$"),
    ]:
        for task in task_order:
            marker, color = styles[task]
            subset = [row for row in rows if row["task"] == task]
            ax.semilogy(
                [row[field] for row in subset],
                [row["k90"] for row in subset],
                linestyle="none",
                marker=marker,
                markersize=2.0,
                color=color,
                markerfacecolor="white" if task in {"transported_local_z", "all_pauli"} else color,
                markeredgewidth=0.55,
                alpha=0.55,
                label=labels[task],
            )
        fit = _linear_fit([row[field] for row in rows], [math.log(row["k90"]) for row in rows])
        xs = np.linspace(min(row[field] for row in rows), max(row[field] for row in rows), 100)
        ys = np.exp(fit["intercept"] + fit["slope"] * xs)
        ax.semilogy(xs, ys, color="#111111", linewidth=0.8, linestyle="--")
        ax.text(
            0.03,
            0.93,
            rf"$R^2={fit['r2']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(axis="both", length=3)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    axes[0].set_ylabel(r"smooth support $K_{0.9}$")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.4, fontsize=4.6)

    ax = axes[2]
    summary_path = DATA_DIR / "r41_fixed_error_budget_summary.csv"
    summary_rows = _read_fixed_error_summary(summary_path) if summary_path.exists() else []
    ratio_tasks = ["local_z", "local_transverse", "hamiltonian_terms"]
    datasets = ["random", "floquet"]
    dataset_labels = {"random": "random", "floquet": "Floquet"}
    task_labels = {"local_z": r"$Z$", "local_transverse": r"$X/Y$", "hamiltonian_terms": r"$H$"}
    width = 0.32
    x0 = np.arange(len(ratio_tasks))
    for offset, dataset in [(-width / 2, "random"), (width / 2, "floquet")]:
        values = [
            next(
                (
                    row["median_ratio"]
                    for row in summary_rows
                    if row["dataset"] == dataset and row["task"] == task
                ),
                math.nan,
            )
            for task in ratio_tasks
        ]
        ax.bar(
            x0 + offset,
            values,
            width=width,
            color="#d9d9d9" if dataset == "random" else "#777777",
            edgecolor="#333333",
            linewidth=0.6,
            label=dataset_labels[dataset],
        )
    ax.axhline(1.0, color="#111111", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1.6)
    ax.set_xticks(x0)
    ax.set_xticklabels([task_labels[task] for task in ratio_tasks])
    ax.set_ylabel("retained-count ratio")
    ax.set_title("fixed-error budget", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.22)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=4.8)
    fig.tight_layout(pad=0.22, w_pad=0.35)
    _save(fig, "fig2_global_task_truncation")


def _plot_predictor_benchmark(rows: list[dict[str, object]]) -> None:
    task_order = [
        "local_z",
        "local_transverse",
        "hamiltonian_terms",
        "transported_local_z",
        "all_pauli",
    ]
    labels = {
        "local_z": "local Z",
        "local_transverse": "local X/Y",
        "hamiltonian_terms": "Hamiltonian",
        "transported_local_z": "transported Z",
        "all_pauli": "all Pauli",
    }
    styles = {
        "local_z": ("o", "#111111"),
        "local_transverse": ("s", "#444444"),
        "hamiltonian_terms": ("D", "#777777"),
        "transported_local_z": ("v", "#999999"),
        "all_pauli": ("^", "#000000"),
    }
    fit_rows = []
    ratio_rows = []
    for subset_name, subset in [
        ("all", rows),
        (
            "n12_density_0.20",
            [
                row
                for row in rows
                if row["n_qubits"] == 12 and abs(row["t_density"] - 0.20) < 1e-9
            ],
        ),
    ]:
        fit_rows.extend(_predictor_fit_rows(subset_name, subset))
        ratio_rows.extend(_predictor_ratio_rows(subset_name, subset, task_order))
    _write_rows(DATA_DIR / "r41_predictor_benchmark_fit.csv", fit_rows)
    _write_rows(DATA_DIR / "r41_predictor_benchmark_ratios.csv", ratio_rows)

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), dpi=300)
    panels = [
        ("global_p2_estimate", _panel_title(0, "Channel average"), r"global $M_2^{\mathrm{op}}$"),
        ("task_p2", _panel_title(1, "Observable resolved"), r"task $\mathcal{P}_2(U;O)$"),
    ]
    for ax, (field, title, xlabel) in zip(axes[:2], panels):
        for task in task_order:
            marker, color = styles[task]
            subset = [row for row in rows if row["task"] == task]
            ax.semilogy(
                [row[field] for row in subset],
                [row["k90"] for row in subset],
                linestyle="none",
                marker=marker,
                markersize=2.0,
                color=color,
                markerfacecolor="white" if task in {"transported_local_z", "all_pauli"} else color,
                markeredgewidth=0.55,
                alpha=0.58,
                label=labels[task],
            )
        fit = _linear_fit([row[field] for row in rows], [math.log(row["k90"]) for row in rows])
        xs = np.linspace(
            min(row[field] for row in rows),
            max(row[field] for row in rows),
            100,
        )
        ys = np.exp(fit["intercept"] + fit["slope"] * xs)
        ax.semilogy(xs, ys, color="#111111", linewidth=0.8, linestyle="--")
        ax.text(
            0.03,
            0.93,
            rf"$R^2={fit['r2']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(axis="both", length=3)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    axes[0].set_ylabel(r"smooth support $K_{0.9}$")

    ax = axes[2]
    ratio_subset = [
        row
        for row in ratio_rows
        if row["subset"] == "all" and row["task"] != "all_pauli"
    ]
    ratio_task_order = [
        "local_z",
        "local_transverse",
        "hamiltonian_terms",
        "transported_local_z",
    ]
    data = [
        [float(row["k90_over_all_pauli"]) for row in ratio_subset if row["task"] == task]
        for task in ratio_task_order
    ]
    ax.boxplot(
        data,
        positions=np.arange(len(ratio_task_order)) + 1,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.9},
        boxprops={"facecolor": "#e6e6e6", "edgecolor": "#555555", "linewidth": 0.7},
        whiskerprops={"color": "#555555", "linewidth": 0.7},
        capprops={"color": "#555555", "linewidth": 0.7},
    )
    ax.axhline(1.0, color="#111111", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_{0.9}(O)/K_{0.9}(\mathrm{all})$")
    ax.set_xticks(np.arange(len(ratio_task_order)) + 1)
    ax.set_xticklabels([labels[task] for task in ratio_task_order], rotation=32, ha="right")
    ax.set_title(_panel_title(2, "Same-circuit ratio"), fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.22)
    _finish_with_legend(
        fig,
        axes,
        ncol=5,
        bottom=0.12,
        pad=0.25,
        w_pad=0.35,
        legend_y=0.030,
    )
    _save(fig, "fig4_predictor_benchmark")


def _predictor_fit_rows(subset_name: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    y = [math.log(row["k90"]) for row in rows]
    for predictor, field in [
        ("global_operator_sre", "global_p2_estimate"),
        ("task_entropy", "task_p2"),
    ]:
        x = [row[field] for row in rows]
        fit = _linear_fit(x, y)
        output.append(
            {
                "subset": subset_name,
                "predictor": predictor,
                "pairs": len(rows),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "coefficient_of_determination": fit["r2"],
                "pearson_correlation": fit["pearson"],
            }
        )
    return output


def _predictor_ratio_rows(
    subset_name: str,
    rows: list[dict[str, object]],
    task_order: list[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, float, int, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (
            row["n_qubits"],
            row["layers"],
            row["t_density"],
            row["repeat"],
            int(row["circuit_seed"]),
        )
        groups.setdefault(key, {})[str(row["task"])] = row
    output = []
    for key, by_task in groups.items():
        if "all_pauli" not in by_task:
            continue
        all_k90 = float(by_task["all_pauli"]["k90"])
        for task in task_order:
            if task not in by_task:
                continue
            row = by_task[task]
            output.append(
                {
                    "subset": subset_name,
                    "n_qubits": key[0],
                    "layers": key[1],
                    "t_density": key[2],
                    "repeat": key[3],
                    "task": task,
                    "k90": row["k90"],
                    "all_pauli_k90": all_k90,
                    "k90_over_all_pauli": float(row["k90"]) / max(all_k90, 1e-300),
                }
            )
    return output


def _linear_fit(x_values: list[float], y_values: list[float]) -> dict[str, float]:
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


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_estimator_collapse(
    rows: list[dict[str, object]],
    truncation_rows: list[dict[str, object]],
    observable_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0), dpi=300)
    ax = axes[0]
    markers = ["o", "s", "^", "D", "v", "P"]
    cases = sorted({row["case"] for row in rows})
    for marker, case in zip(markers, cases):
        subset = sorted([row for row in rows if row["case"] == case], key=lambda row: row["effective_samples"])
        ax.loglog(
            [row["effective_samples"] for row in subset],
            [row["relative_rmse"] for row in subset],
            linestyle="none",
            marker=marker,
            markersize=3.0,
            color="#555555",
            markerfacecolor="white" if "all" in case or "transported" in case else "#555555",
            markeredgewidth=0.8,
            label=case,
        )
    xmin = min(row["effective_samples"] for row in rows)
    xmax = max(row["effective_samples"] for row in rows)
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    ax.loglog(xs, xs ** -0.5, color="#111111", linewidth=0.9, linestyle="--", label=r"$x^{-1/2}$")
    ax.set_xlabel(r"effective samples $m e^{-\mathcal{P}_2}$")
    ax.set_ylabel("relative RMSE")
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=4.8)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    ax.set_title("terminal sampling", fontsize=7, pad=2)

    ax = axes[1]
    if truncation_rows:
        for row in truncation_rows:
            marker = "s" if "all Pauli" in row["case"] else "o"
            face = "white" if "transported" in row["case"] or "all Pauli" in row["case"] else "#555555"
            ax.loglog(
                [row["exp_task_p2"]],
                [row["k90"]],
                linestyle="none",
                marker=marker,
                markersize=3.6,
                color="#555555",
                markerfacecolor=face,
                markeredgewidth=0.8,
            )
        xmin = min(row["exp_task_p2"] for row in truncation_rows)
        xmax = max(row["exp_task_p2"] for row in truncation_rows)
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        ax.loglog(xs, xs, color="#777777", linewidth=0.8, linestyle=":", label=r"$e^{\mathcal{P}_2}$")
        ax.loglog(xs, 0.81 * xs, color="#111111", linewidth=0.9, linestyle="--", label=r"$0.9^2e^{\mathcal{P}_2}$")
        ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=4.8)
        ax.set_xticks([2, 3, 4, 6])
        ax.set_xticklabels(["2", "3", "4", "6"])
    ax.set_xlabel(r"$e^{\mathcal{P}_2}$")
    ax.set_ylabel(r"$K_{0.9}$")
    ax.set_title("terminal truncation", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.22)

    ax = axes[2]
    if observable_rows:
        subset = [
            row
            for row in observable_rows
            if row["state_response"] == "tilted_product" and row["exact_rms"] > 1e-10
        ]
        xs_raw = [
            row["avg_retained_terms"] / max(np.exp(row["task_p2"]), 1e-12)
            for row in subset
        ]
        ax.semilogy(
            xs_raw,
            [max(row["relative_rms_error"], 1e-12) for row in subset],
            linestyle="none",
            marker="o",
            markersize=2.2,
            color="#777777",
            alpha=0.35,
            markeredgewidth=0.0,
        )
        masses = sorted({row["target_mass"] for row in subset})
        means = []
        mean_xs = []
        for mass in masses:
            rows_for_mass = [row for row in subset if row["target_mass"] == mass]
            values = [max(row["relative_rms_error"], 1e-12) for row in rows_for_mass]
            mean_xs.append(
                float(
                    np.mean(
                        [
                            row["avg_retained_terms"] / max(np.exp(row["task_p2"]), 1e-12)
                            for row in rows_for_mass
                        ]
                    )
                )
            )
            means.append(float(np.mean(values)))
        ax.semilogy(
            mean_xs,
            means,
            color="#111111",
            marker="o",
            markersize=3.0,
            linewidth=0.9,
            linestyle="-",
            label="tilted product",
        )
        ax.set_ylim(5e-3, 8e-1)
    ax.set_xlabel(r"budget ratio $B/e^{\mathcal{P}_2}$")
    ax.set_ylabel("relative RMS error")
    ax.set_title("observable truncation", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=4.8)

    fig.tight_layout(pad=0.25, w_pad=0.35)
    _save(fig, "fig3_operational_tests")
    _save(fig, "fig4_estimator_collapse")
    summary_path = DATA_DIR / "r41_fixed_error_budget_summary.csv"
    fixed_error_rows = _read_fixed_error_summary(summary_path) if summary_path.exists() else []
    _plot_estimator_collapse_main(rows, observable_rows, fixed_error_rows)


def _plot_estimator_collapse_main(
    rows: list[dict[str, object]],
    observable_rows: list[dict[str, object]],
    fixed_error_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.85), dpi=300)
    ax = axes[0]
    markers = ["o", "s", "^", "D", "v", "P"]
    cases = sorted({row["case"] for row in rows})
    for marker, case in zip(markers, cases):
        subset = sorted(
            [row for row in rows if row["case"] == case],
            key=lambda row: row["effective_samples"],
        )
        ax.loglog(
            [row["effective_samples"] for row in subset],
            [row["relative_rmse"] for row in subset],
            linestyle="none",
            marker=marker,
            markersize=2.4,
            color="#555555",
            markerfacecolor="white" if "all" in case or "transported" in case else "#555555",
            markeredgewidth=0.65,
        )
    xmin = min(row["effective_samples"] for row in rows)
    xmax = max(row["effective_samples"] for row in rows)
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    ax.loglog(xs, xs ** -0.5, color="#111111", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$m e^{-\mathcal{P}_2}$")
    ax.set_ylabel("relative RMSE")
    ax.tick_params(axis="both", length=2.5)
    ax.grid(True, which="both", linewidth=0.25, alpha=0.22)
    ax.set_title("sampling", fontsize=7, pad=2)

    ax = axes[1]
    if observable_rows:
        subset = [
            row
            for row in observable_rows
            if row["state_response"] == "tilted_product" and row["exact_rms"] > 1e-10
        ]
        curve_styles = {
            "local X/Y n8": ("local $X/Y$", "s", "#4d4d4d"),
            "Hamiltonian n10": ("Hamiltonian", "D", "#8a8a8a"),
            "all Pauli n12": ("all Pauli", "^", "#111111"),
        }
        for case, (label, marker, color) in curve_styles.items():
            case_rows = sorted(
                [row for row in subset if row["case"] == case],
                key=lambda row: row["target_mass"],
            )
            if not case_rows:
                continue
            ax.semilogy(
                [row["target_mass"] for row in case_rows],
                [max(row["relative_rms_error"], 1e-12) for row in case_rows],
                color=color,
                marker=marker,
                markersize=2.7,
                linewidth=0.85,
                linestyle="-",
                markerfacecolor="white" if "all Pauli" in case else color,
                markeredgewidth=0.6,
                label=label,
            )
        ax.set_ylim(5e-5, 8e-1)
        ax.legend(frameon=False, loc="lower left", handlelength=1.3, fontsize=4.8)
    ax.set_xlabel(r"retained mass $\eta$")
    ax.set_ylabel("relative RMS error")
    ax.set_title("truncation", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=2.5)
    ax.grid(True, which="both", linewidth=0.25, alpha=0.22)
    ax = axes[2]
    if fixed_error_rows:
        dataset_order = ["random", "floquet"]
        task_order = ["local_z", "local_transverse", "hamiltonian_terms"]
        dataset_labels = {"random": "Rand.", "floquet": "Floq."}
        task_labels = {"local_z": "$Z$", "local_transverse": "$X/Y$", "hamiltonian_terms": "$H$"}
        colors = {"local_z": "#111111", "local_transverse": "#777777", "hamiltonian_terms": "#c0c0c0"}
        lookup = {(row["dataset"], row["task"]): row for row in fixed_error_rows}
        width = 0.24
        x_positions = np.arange(len(dataset_order), dtype=float)
        for offset, task in zip([-width, 0.0, width], task_order):
            medians = []
            lower = []
            upper = []
            for dataset in dataset_order:
                row = lookup.get((dataset, task))
                if row is None:
                    medians.append(np.nan)
                    lower.append(0.0)
                    upper.append(0.0)
                    continue
                med = row["median_ratio"]
                medians.append(med)
                lower.append(max(med - row["q25_ratio"], 0.0))
                upper.append(max(row["q75_ratio"] - med, 0.0))
            ax.bar(
                x_positions + offset,
                medians,
                width=width,
                color=colors[task],
                edgecolor="#111111",
                linewidth=0.4,
                label=task_labels[task],
            )
            ax.errorbar(
                x_positions + offset,
                medians,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#111111",
                elinewidth=0.45,
                capsize=1.2,
                capthick=0.45,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([dataset_labels[item] for item in dataset_order])
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, loc="upper left", ncol=3, columnspacing=0.6, handlelength=1.0)
    ax.set_ylabel("ratio")
    ax.set_title("termwise fixed-error budget", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=2.5)
    ax.grid(True, axis="y", linewidth=0.25, alpha=0.22)

    ax = axes[3]
    ablation_path = DATA_DIR / "r41_predictor_ablation.csv"
    if ablation_path.exists():
        ablation_rows = _read_predictor_ablation(ablation_path)
        order = ["global_operator_sre", "t_count", "active_t_count", "task_p2"]
        short_labels = {
            "global_operator_sre": r"$M_2$",
            "t_count": r"$T$",
            "active_t_count": r"$T_{\rm cone}$",
            "task_p2": r"$\mathcal{P}_2$",
        }
        lookup = {row["predictor"]: row for row in ablation_rows if row["target"] == "log_k90"}
        values = [lookup[item]["coefficient_of_determination"] for item in order if item in lookup]
        labels = [short_labels[item] for item in order if item in lookup]
        x_values = np.arange(len(values))
        ax.bar(
            x_values,
            values,
            color=["#d8d8d8", "#bdbdbd", "#8d8d8d", "#111111"][: len(values)],
            edgecolor="#111111",
            linewidth=0.45,
        )
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        for index, value in enumerate(values):
            ax.text(
                index,
                min(value + 0.035, 0.97),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.2,
                color="#111111",
            )
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"predicting $\log K_{0.9}$", fontsize=7, pad=2)
    ax.tick_params(axis="both", length=2.5)
    ax.grid(True, axis="y", linewidth=0.25, alpha=0.22)

    fig.tight_layout(pad=0.15, w_pad=0.30)
    _save(fig, "fig3_operational_tests_main")


def _plot_mqt_benchmark(rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if row["t_count"] > 0 and row["k90"] > 0]
    if not rows:
        return
    task_order = ["local_z", "local_transverse", "hamiltonian_terms", "all_pauli"]
    core_rows = [row for row in rows if row["task"] in set(task_order)]
    labels = {
        "local_z": "local Z",
        "local_transverse": "local X/Y",
        "hamiltonian_terms": "Hamiltonian",
        "all_pauli": "all Pauli",
    }
    styles = {
        "local_z": ("o", "#111111"),
        "local_transverse": ("s", "#444444"),
        "hamiltonian_terms": ("D", "#777777"),
        "all_pauli": ("^", "#000000"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), dpi=300)
    for ax, field, xlabel, title in [
        (
            axes[0],
            "global_p2_estimate",
            r"global $M_2^{\mathrm{op}}$ estimate",
            _panel_title(0, "Channel average"),
        ),
        (
            axes[1],
            "task_p2",
            r"task $\mathcal{P}_2(U;O)$",
            _panel_title(1, "Observable resolved"),
        ),
    ]:
        for task in task_order:
            marker, color = styles[task]
            subset = [row for row in core_rows if row["task"] == task]
            ax.semilogy(
                [row[field] for row in subset],
                [row["k90"] for row in subset],
                linestyle="none",
                marker=marker,
                markersize=3.0,
                color=color,
                markerfacecolor="white" if task == "all_pauli" else color,
                markeredgewidth=0.65,
                alpha=0.72,
                label=labels[task],
            )
        fit = _linear_fit([row[field] for row in core_rows], [math.log(row["k90"]) for row in core_rows])
        xs = np.linspace(min(row[field] for row in core_rows), max(row[field] for row in core_rows), 100)
        ys = np.exp(fit["intercept"] + fit["slope"] * xs)
        ax.semilogy(xs, ys, color="#111111", linewidth=0.8, linestyle="--")
        ax.text(
            0.03,
            0.93,
            rf"$R^2={fit['r2']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(axis="both", length=3)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.22)
    axes[0].set_ylabel(r"smooth support $K_{0.9}$")

    by_benchmark: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in core_rows:
        by_benchmark.setdefault((row["benchmark"], row["size"]), {})[row["task"]] = row
    ratio_rows = []
    for group in by_benchmark.values():
        if "all_pauli" not in group:
            continue
        all_k90 = float(group["all_pauli"]["k90"])
        for task in task_order:
            if task == "all_pauli" or task not in group:
                continue
            ratio_rows.append({"task": task, "ratio": float(group[task]["k90"]) / all_k90})
    ax = axes[2]
    ratio_order = ["local_z", "local_transverse", "hamiltonian_terms"]
    data = [[row["ratio"] for row in ratio_rows if row["task"] == task] for task in ratio_order]
    ax.boxplot(
        data,
        positions=np.arange(len(ratio_order)) + 1,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.9},
        boxprops={"facecolor": "#e6e6e6", "edgecolor": "#555555", "linewidth": 0.7},
        whiskerprops={"color": "#555555", "linewidth": 0.7},
        capprops={"color": "#555555", "linewidth": 0.7},
    )
    ax.axhline(1.0, color="#111111", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_{0.9}(O)/K_{0.9}(\mathrm{all})$")
    ax.set_xticks(np.arange(len(ratio_order)) + 1)
    ax.set_xticklabels([labels[task] for task in ratio_order], rotation=32, ha="right")
    ax.set_title(_panel_title(2, "Same-circuit ratio"), fontsize=7, pad=2)
    ax.tick_params(axis="both", length=3)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.22)
    _finish_with_legend(
        fig,
        axes,
        ncol=4,
        bottom=0.12,
        pad=0.25,
        w_pad=0.35,
        fontsize=4.8,
        legend_y=0.030,
    )
    _save(fig, "fig5_mqt_benchmark")


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(RESULTS_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(RESULTS_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
