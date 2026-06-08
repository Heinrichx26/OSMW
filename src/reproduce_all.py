from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


QUICK_STEPS = [
    ["src/r41_quick_validation.py"],
    ["src/r41_observable_cost.py", "--validate"],
    ["src/r41_large_n_separation.py", "--quick"],
    ["src/r41_observable_dark_gap.py", "--quick"],
    ["src/r41_fixed_error_budget.py", "--validate"],
    ["src/r41_clifford_covariance_checks.py", "--quick"],
    ["src/r41_proxy_counterexamples.py", "--validate"],
    ["src/r41_task_calculus_checks.py", "--quick"],
    ["src/r41_rotation_action_checks.py", "--quick"],
    ["src/r41_stress_workloads.py", "--quick"],
    ["src/r41_2d_stress_workloads.py", "--quick"],
    ["src/r41_estimation_workload_benchmark.py", "--quick"],
]

MQT_QUICK_STEPS = [
    ["src/r41_benchmark_validation.py", "--validate"],
]

FULL_STEPS = [
    ["src/r41_quick_validation.py"],
    ["src/r41_pauli_path_checks.py"],
    ["src/r41_family_checks.py"],
    ["src/r41_enumerate_small_circuits.py"],
    ["src/r41_task_resolved_checks.py"],
    ["src/r41_pauli_scaling.py"],
    ["src/r41_observable_cost.py"],
    ["src/r41_large_n_separation.py"],
    ["src/r41_observable_dark_gap.py"],
    ["src/r41_predictor_ablation.py"],
    ["src/r41_clifford_covariance_checks.py"],
    ["src/r41_proxy_counterexamples.py"],
    ["src/r41_task_calculus_checks.py"],
    ["src/r41_rotation_action_checks.py"],
    ["src/r41_stress_workloads.py"],
    ["src/r41_2d_stress_workloads.py"],
    ["src/r41_estimation_workload_benchmark.py"],
    ["src/r41_resampling_stats.py"],
]

MQT_FULL_STEPS = [
    ["src/r41_benchmark_validation.py"],
    ["src/r41_fixed_error_budget.py"],
]

POST_QUICK_STEPS = [
    ["src/r41_same_support_separation.py", "--quick"],
    ["src/r41_weighted_observable_validation.py", "--quick"],
]

POST_FULL_STEPS = [
    ["src/r41_decision_validations.py"],
    ["src/r41_same_support_separation.py"],
    ["src/r41_weighted_observable_validation.py"],
]

PLOT_STEPS = [
    ["src/plot_r41.py"],
    ["src/r41_stress_workloads.py", "--plot-only"],
    ["src/r41_2d_stress_workloads.py", "--plot-only"],
    ["src/r41_estimation_workload_benchmark.py", "--plot-only"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run validation-only checks")
    parser.add_argument("--skip-mqt", action="store_true", help="skip MQT Bench scripts")
    parser.add_argument("--skip-plots", action="store_true", help="skip figure regeneration")
    args = parser.parse_args()

    quick = args.quick
    steps = list(QUICK_STEPS if quick else FULL_STEPS)
    if not args.skip_mqt:
        steps.extend(MQT_QUICK_STEPS if quick else MQT_FULL_STEPS)
    steps.extend(POST_QUICK_STEPS if quick else POST_FULL_STEPS)
    if not quick and not args.skip_plots:
        steps.extend(PLOT_STEPS)

    for step in steps:
        print("$", " ".join(["python", *step]), flush=True)
        subprocess.run([sys.executable, *step], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
