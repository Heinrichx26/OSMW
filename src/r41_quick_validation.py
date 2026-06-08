from __future__ import annotations

import json
import math
from pathlib import Path

from r41_path_entropy import (
    brickwork_clifford_family,
    gate,
    linear_fit,
    operator_magic_from_gates,
    path_summary,
    simple_clifford_path_family,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_t_magic = math.log(4.0 / 3.0)
    checks: list[dict[str, object]] = []

    h_magic = operator_magic_from_gates(1, [gate("H", 0)])
    checks.append(_check_close("H-only operator magic", h_magic, 0.0, 1e-10))

    t_magic = operator_magic_from_gates(1, [gate("T", 0)])
    checks.append(_check_close("T operator magic", t_magic, expected_t_magic, 1e-10))

    ht_cnot_magic = operator_magic_from_gates(
        2,
        [gate("H", 0), gate("T", 0), gate("CNOT", 0, 1)],
    )
    checks.append(
        _check_close("H+T+CNOT operator magic", ht_cnot_magic, expected_t_magic, 1e-10)
    )

    t_paths = path_summary(1, [gate("T", 0)])
    checks.append(_check_close("T path C2", t_paths.median_c2, 0.0, 1e-12))

    simple_depths = [1, 2, 3, 4]
    simple_c2 = []
    for depth in simple_depths:
        summary = path_summary(
            2,
            simple_clifford_path_family(2, depth),
            max_total_paths=1_000_000,
        )
        simple_c2.append(summary.median_c2)
    simple_fit = linear_fit(simple_depths, simple_c2)
    checks.append(
        {
            "name": "simple Clifford path growth",
            "value": simple_fit["r2"],
            "expected": ">= 0.98",
            "tolerance": None,
            "passed": simple_fit["r2"] >= 0.98 and simple_fit["slope"] > 0,
        }
    )

    simple_magic = operator_magic_from_gates(2, simple_clifford_path_family(2, 4))
    checks.append(_check_close("simple Clifford family magic", simple_magic, 0.0, 1e-10))

    brickwork_depths = [1, 2, 3, 4]
    brickwork_c2 = []
    for depth in brickwork_depths:
        summary = path_summary(
            2,
            brickwork_clifford_family(2, depth),
            max_total_paths=1_000_000,
        )
        brickwork_c2.append(summary.median_c2)
    brickwork_fit = linear_fit(brickwork_depths, brickwork_c2)
    brickwork_magic = operator_magic_from_gates(2, brickwork_clifford_family(2, 4))

    output = {
        "expected_t_magic_log_4_over_3": expected_t_magic,
        "checks": checks,
        "path_summaries": {
            "T": t_paths.to_dict(),
            "simple_clifford_n2": {
                "depths": simple_depths,
                "median_c2": simple_c2,
                "fit": simple_fit,
                "operator_magic_depth_4": simple_magic,
            },
            "brickwork_clifford_n2": {
                "depths": brickwork_depths,
                "median_c2": brickwork_c2,
                "fit": brickwork_fit,
                "operator_magic_depth_4": brickwork_magic,
            },
        },
    }

    output_path = RESULTS_DIR / "r41_quick_validation_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    failed = [item for item in checks if not item["passed"]]
    if failed:
        print(json.dumps(output, indent=2))
        raise SystemExit(f"{len(failed)} quick validation check(s) failed")

    print(json.dumps(output, indent=2))


def _check_close(name: str, value: float, expected: float, tolerance: float) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "expected": expected,
        "tolerance": tolerance,
        "passed": abs(value - expected) <= tolerance,
    }


if __name__ == "__main__":
    main()
