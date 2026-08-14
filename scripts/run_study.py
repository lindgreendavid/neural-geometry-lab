"""Execute the frozen v1.0 study and build browser artifacts."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from neural_geometry_lab.data import class_counts, construct_condition, load_official_splits
from neural_geometry_lab.experiment import (
    CONDITIONS,
    SEEDS,
    fit_logistic_reference,
    train_neural_run,
)
from neural_geometry_lab.metrics import spearman

ROOT = Path(__file__).resolve().parents[1]


def run_one(specification: tuple[str, int]) -> tuple[str, int, dict[str, Any], Any, Any]:
    """Run one independent, deterministic condition-seed cell."""
    name, seed = specification
    train, test = load_official_splits(ROOT / "data" / "raw")
    condition = construct_condition(train, name)
    run = train_neural_run(condition, test, seed, keep_display=seed == 1001)
    return name, seed, run.summary, run.checkpoints, run.display_states


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize(neural_runs: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [run for run in neural_runs if run["condition"] == "clean"]
    long_tail = [run for run in neural_runs if run["condition"] == "long_tail"]
    clean_by_seed = {run["seed"]: run for run in clean}
    tail_by_seed = {run["seed"]: run for run in long_tail}

    def direction_count(metric: str, *, no_greater: bool = False) -> int:
        count = 0
        for run in clean:
            first = run["first_zero"]
            if first is None:
                continue
            final_value = float(run["final"]["neural_collapse"][metric])
            first_value = float(first["neural_collapse"][metric])
            count += int(final_value <= first_value if no_greater else final_value < first_value)
        return count

    imbalance_count = sum(
        float(tail_by_seed[seed]["final"]["neural_collapse"]["nc2_simplex_deviation"])
        > float(clean_by_seed[seed]["final"]["neural_collapse"]["nc2_simplex_deviation"])
        for seed in SEEDS
    )
    final_nc1 = np.asarray(
        [float(run["final"]["neural_collapse"]["nc1_variability"]) for run in neural_runs]
    )
    final_nc2 = np.asarray(
        [float(run["final"]["neural_collapse"]["nc2_simplex_deviation"]) for run in neural_runs]
    )
    test_accuracy = np.asarray([float(run["final"]["test"]["accuracy"]) for run in neural_runs])
    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        runs = [run for run in neural_runs if run["condition"] == condition]
        condition_summary[condition] = {
            "median_test_accuracy": median(
                [float(run["final"]["test"]["accuracy"]) for run in runs]
            ),
            "range_test_accuracy": [
                min(float(run["final"]["test"]["accuracy"]) for run in runs),
                max(float(run["final"]["test"]["accuracy"]) for run in runs),
            ],
            "median_nc1": median(
                [float(run["final"]["neural_collapse"]["nc1_variability"]) for run in runs]
            ),
            "median_nc2": median(
                [float(run["final"]["neural_collapse"]["nc2_simplex_deviation"]) for run in runs]
            ),
            "zero_error_seeds": sum(run["first_zero_epoch"] is not None for run in runs),
        }
    return {
        "gates": {
            "clean_nc1_after_zero": {
                "count": direction_count("nc1_variability"),
                "passed": direction_count("nc1_variability") >= 8,
            },
            "clean_nc2_after_zero": {
                "count": direction_count("nc2_simplex_deviation"),
                "passed": direction_count("nc2_simplex_deviation") >= 8,
            },
            "clean_nc4_after_zero": {
                "count": direction_count("nc4_disagreement", no_greater=True),
                "passed": direction_count("nc4_disagreement", no_greater=True) >= 8,
            },
            "imbalance_worsens_nc2": {
                "count": int(imbalance_count),
                "passed": imbalance_count >= 8,
            },
        },
        "condition_summary": condition_summary,
        "descriptive_correlations": {
            "spearman_final_nc1_vs_test_accuracy": spearman(final_nc1, test_accuracy),
            "spearman_final_nc2_vs_test_accuracy": spearman(final_nc2, test_accuracy),
        },
    }


def main() -> None:
    train, test = load_official_splits(ROOT / "data" / "raw")
    conditions = {name: construct_condition(train, name) for name in CONDITIONS}
    logistic = [fit_logistic_reference(conditions[name], test) for name in CONDITIONS]
    neural_summaries: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}
    browser_states: dict[str, Any] = {}
    specifications = [(name, seed) for name in CONDITIONS for seed in SEEDS]
    workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, spec): spec for spec in specifications}
        for future in as_completed(futures):
            name, seed, summary, checkpoints, display_states = future.result()
            print(f"completed {name} seed {seed}", flush=True)
            neural_summaries.append(summary)
            if seed == 1001:
                trajectories[name] = checkpoints
                browser_states[name] = display_states
    condition_order = {name: index for index, name in enumerate(CONDITIONS)}
    neural_summaries.sort(key=lambda run: (condition_order[run["condition"]], run["seed"]))
    results = {
        "schema_version": "1.0.0",
        "product_version": "1.0.0",
        "protocol": "docs/protocol-v1.0.md",
        "dataset": {
            "train_rows": int(train.labels.size),
            "test_rows": int(test.labels.size),
            "train_class_counts": class_counts(train.labels),
            "test_class_counts": class_counts(test.labels),
            "conditions": {
                name: {
                    "rows": int(condition.observed_labels.size),
                    "observed_class_counts": class_counts(condition.observed_labels),
                    "true_class_counts": class_counts(condition.true_labels),
                    "changed_labels": condition.changed_labels,
                }
                for name, condition in conditions.items()
            },
        },
        "logistic_regression": logistic,
        "neural_runs": neural_summaries,
    }
    results["evaluation"] = summarize(neural_summaries)
    report_path = ROOT / "reports" / "results-v1.0.json"
    report_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    trajectory_path = ROOT / "reports" / "trajectory-v1.0.json"
    trajectory_path.write_text(
        json.dumps(trajectories, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    site = {
        "summary": results["evaluation"],
        "dataset": results["dataset"],
        "logistic_regression": logistic,
        "trajectories": trajectories,
        "display_states": browser_states,
    }
    site_path = ROOT / "site" / "data" / "study-v1.0.json"
    site_path.write_text(
        json.dumps(site, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results["evaluation"], indent=2))


if __name__ == "__main__":
    main()
