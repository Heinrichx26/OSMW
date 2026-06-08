from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from r41_observable_cost import Label, build_terminal_columns, terminal_truncation_profile
from r41_path_entropy import Gate, gate, operator_magic_from_gates
from r41_predictor_ablation import active_t_count_for_label


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--max-support", type=int, default=12)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_counterexample()
    if args.validate:
        (DATA_DIR / "r41_proxy_counterexamples_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        print(json.dumps({"validation": validation}, indent=2))
        if not validation["passed"]:
            raise SystemExit(1)
        return

    rows = run_counterexamples(args.max_support)
    summary = summarize(rows)
    payload = {
        "validation": validation,
        "summary": summary,
        "rows": rows,
        "interpretation": {
            "purpose": "stress test for global M2, gate-list non-Clifford count, and selected-light-cone T-count proxies",
            "collapse_family": "nonadjacent Clifford-return T echo on measured qubits plus magic spectators",
            "spread_family": "T-H-T blocks on measured qubits plus nonadjacent Clifford-return padding spectators",
            "matched_quantities": [
                "operator_magic_analytic",
                "gate_list_nonclifford_count",
                "task_active_t_count",
                "task_support_qubits",
            ],
            "separating_quantities": ["task_p2", "k90", "terminal_support"],
        },
    }
    write_csv(DATA_DIR / "r41_proxy_counterexamples.csv", rows)
    (DATA_DIR / "r41_proxy_counterexamples.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    plot_rows(rows)
    print(json.dumps({"validation": validation, "summary": summary}, indent=2))


def validate_counterexample() -> dict[str, object]:
    support = 1
    n_qubits = 3 * support
    task = task_label(support)
    high = spread_circuit(support)
    low = collapse_circuit(support)
    high_profile = profile(n_qubits, high, task)
    low_profile = profile(n_qubits, low, task)
    exact_high_magic = operator_magic_from_gates(n_qubits, high)
    exact_low_magic = operator_magic_from_gates(n_qubits, low)
    expected_magic = analytic_operator_magic(support)
    checks = [
        {
            "case": "matched_operator_magic_high",
            "observed": exact_high_magic,
            "expected": expected_magic,
            "abs_error": abs(exact_high_magic - expected_magic),
        },
        {
            "case": "matched_operator_magic_low",
            "observed": exact_low_magic,
            "expected": expected_magic,
            "abs_error": abs(exact_low_magic - expected_magic),
        },
        {
            "case": "matched_gate_list_nonclifford_count",
            "observed": gate_list_nonclifford_count(high),
            "expected": gate_list_nonclifford_count(low),
            "abs_error": abs(
                gate_list_nonclifford_count(high) - gate_list_nonclifford_count(low)
            ),
        },
        {
            "case": "matched_task_active_t_count",
            "observed": active_t_count_for_label(n_qubits, high, task),
            "expected": active_t_count_for_label(n_qubits, low, task),
            "abs_error": abs(
                active_t_count_for_label(n_qubits, high, task)
                - active_t_count_for_label(n_qubits, low, task)
            ),
        },
        {
            "case": "collapse_k90",
            "observed": low_profile["k90"],
            "expected": 1,
            "abs_error": abs(int(low_profile["k90"]) - 1),
        },
        {
            "case": "spread_k90",
            "observed": high_profile["k90"],
            "expected": 3,
            "abs_error": abs(int(high_profile["k90"]) - 3),
        },
    ]
    max_error = max(float(item["abs_error"]) for item in checks)
    return {"checks": checks, "max_abs_error": max_error, "passed": max_error < 1e-10}


def run_counterexamples(max_support: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for support in range(1, max_support + 1):
        n_qubits = 3 * support
        task = task_label(support)
        for family, gates in [
            ("collapse_echo", collapse_circuit(support)),
            ("spread_tht", spread_circuit(support)),
        ]:
            terminal = profile(n_qubits, gates, task)
            rows.append(
                {
                    "support_qubits": support,
                    "n_qubits": n_qubits,
                    "family": family,
                    "operator_magic_analytic": analytic_operator_magic(support),
                    "gate_list_nonclifford_count": gate_list_nonclifford_count(gates),
                    "task_active_t_count": active_t_count_for_label(n_qubits, gates, task),
                    "task_p2": terminal["task_p2"],
                    "exp_task_p2": terminal["exp_task_p2"],
                    "terminal_support": terminal["terminal_support"],
                    "k50": terminal["k50"],
                    "k90": terminal["k90"],
                    "k95": terminal["k95"],
                    "avg_terminal_terms": terminal["avg_terminal_terms"],
                }
            )
    return add_pair_ratios(rows)


def collapse_circuit(support: int) -> list[Gate]:
    gates: list[Gate] = []
    for q in range(support):
        gates.append(gate("T", q))
        gates.append(gate("H", q))
        gates.append(gate("S", q))
        gates.append(gate("S", q))
        gates.append(gate("H", q))
        gates.append(gate("T", q))
    for q in range(support, 3 * support):
        gates.append(gate("T", q))
    return gates


def spread_circuit(support: int) -> list[Gate]:
    gates: list[Gate] = []
    for q in range(support):
        gates.append(gate("T", q))
        gates.append(gate("H", q))
        gates.append(gate("T", q))
    for q in range(support, 2 * support):
        gates.append(gate("T", q))
        gates.append(gate("S", q))
        gates.append(gate("T", q))
        gates.append(gate("S", q))
    return gates


def task_label(support: int) -> Label:
    return ((1 << support) - 1, 0)


def profile(n_qubits: int, gates: list[Gate], label: Label) -> dict[str, object]:
    terminal = build_terminal_columns(n_qubits, gates, [label], max_terms=2_000_000)
    return terminal_truncation_profile(terminal)


def analytic_operator_magic(support: int) -> float:
    return 2.0 * support * math.log(4.0 / 3.0)


def gate_list_nonclifford_count(gates: list[Gate]) -> int:
    return sum(1 for item in gates if item.name in {"T", "TDG"})


def add_pair_ratios(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_support: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_support.setdefault(int(row["support_qubits"]), {})[str(row["family"])] = row
    for support, pair in by_support.items():
        low = pair["collapse_echo"]
        high = pair["spread_tht"]
        low_k90 = float(low["k90"])
        high_k90 = float(high["k90"])
        low_p2 = float(low["task_p2"])
        high_p2 = float(high["task_p2"])
        for row in pair.values():
            row["spread_to_collapse_k90_ratio"] = high_k90 / low_k90
            row["spread_minus_collapse_task_p2"] = high_p2 - low_p2
            row["matched_pair_id"] = f"support_{support}"
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    supports = sorted({int(row["support_qubits"]) for row in rows})
    final_support = supports[-1]
    final_rows = [row for row in rows if int(row["support_qubits"]) == final_support]
    high = next(row for row in final_rows if row["family"] == "spread_tht")
    low = next(row for row in final_rows if row["family"] == "collapse_echo")
    matched = all(
        high[key] == low[key]
        for key in [
            "operator_magic_analytic",
            "gate_list_nonclifford_count",
            "task_active_t_count",
        ]
    )
    return {
        "max_support_qubits": final_support,
        "matched_global_and_proxy_quantities": matched,
        "operator_magic": high["operator_magic_analytic"],
        "gate_list_nonclifford_count": high["gate_list_nonclifford_count"],
        "task_active_t_count": high["task_active_t_count"],
        "collapse_k90": low["k90"],
        "spread_k90": high["k90"],
        "k90_ratio": high["spread_to_collapse_k90_ratio"],
        "collapse_task_p2": low["task_p2"],
        "spread_task_p2": high["task_p2"],
        "task_p2_gap": high["spread_minus_collapse_task_p2"],
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


def plot_rows(rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    for items in by_family.values():
        items.sort(key=lambda row: int(row["support_qubits"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    styles = {
        "collapse_echo": {"label": "echo collapse", "marker": "o", "color": "black"},
        "spread_tht": {"label": "T-H-T spread", "marker": "s", "color": "0.45"},
    }
    for family, items in by_family.items():
        x = [int(row["support_qubits"]) for row in items]
        axes[0].plot(
            x,
            [float(row["k90"]) for row in items],
            linewidth=1.8,
            markersize=4,
            **styles[family],
        )
        axes[1].plot(
            x,
            [float(row["task_p2"]) for row in items],
            linewidth=1.8,
            markersize=4,
            **styles[family],
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("measured support qubits")
    axes[0].set_ylabel(r"$K_{0.9}$")
    axes[0].set_title("(a)Retained support")
    axes[1].set_xlabel("measured support qubits")
    axes[1].set_ylabel(r"$\mathcal{P}_2(U;O)$")
    axes[1].set_title("(b)Task entropy")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.105, 1, 1))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        frameon=False,
        ncol=2,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(
            RESULTS_DIR / f"fig_proxy_counterexamples.{suffix}",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
