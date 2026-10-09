import numpy as np
import pytest

from neural_geometry_lab.within_condition import (
    CONDITIONS,
    build_within_condition,
    coordinate_audit,
)


def synthetic_runs(condition_effect: bool, within_signal: bool, seed: int = 5):
    rng = np.random.default_rng(seed)
    base_accuracy = {"clean": 0.96, "long_tail": 0.92, "label_noise_20": 0.82}
    base_nc = {"clean": 0.2, "long_tail": 0.4, "label_noise_20": 1.2}
    runs = []
    for condition in CONDITIONS:
        for i in range(10):
            noise = rng.normal(0, 0.003)
            accuracy = (base_accuracy[condition] if condition_effect else 0.9) + noise
            nc_value = (base_nc[condition] if condition_effect else 0.5) + (
                -noise * 50 if within_signal else rng.normal(0, 0.02)
            )
            metrics = {
                "nc1_variability": nc_value,
                "nc2_simplex_deviation": nc_value,
                "nc3_self_duality_deviation": nc_value,
                "nc4_disagreement": nc_value,
            }
            runs.append(
                {
                    "condition": condition,
                    "seed": 1001 + i,
                    "final": {"neural_collapse": metrics, "test": {"accuracy": accuracy}},
                }
            )
    return runs


def test_condition_effect_alone_gives_pooled_but_no_within_association():
    out = coordinate_audit(
        synthetic_runs(condition_effect=True, within_signal=False), "nc1_variability"
    )
    assert out["pooled"]["spearman"] < -0.8
    assert abs(out["pooled_within_condition"]["spearman"]) < 0.5
    assert (
        out["pooled_within_condition"]["ci_95"][0] < 0 < out["pooled_within_condition"]["ci_95"][1]
    )


def test_real_within_signal_is_detected_inside_every_condition():
    out = coordinate_audit(
        synthetic_runs(condition_effect=True, within_signal=True), "nc4_disagreement"
    )
    assert out["pooled_within_condition"]["spearman"] < -0.9
    assert all(d["spearman"] < -0.9 for d in out["per_condition"].values())
    assert out["pooled_within_condition"]["ci_95"][1] < -0.5


def test_audit_is_deterministic():
    runs = synthetic_runs(True, False)
    assert coordinate_audit(runs, "nc2_simplex_deviation") == coordinate_audit(
        runs, "nc2_simplex_deviation"
    )


def test_build_reports_all_four_coordinates_and_the_frozen_values():
    results = {
        "neural_runs": synthetic_runs(True, False),
        "evaluation": {"descriptive_correlations": {"spearman_final_nc1_vs_test_accuracy": -0.5}},
    }
    out = build_within_condition(results)
    assert set(out["coordinates"]) == {"nc1", "nc2", "nc3", "nc4"}
    assert out["label"].startswith("POST-HOC")
    assert out["frozen_pooled_spearman"]["spearman_final_nc1_vs_test_accuracy"] == pytest.approx(
        -0.5
    )


def test_constant_metric_yields_nan_intervals_without_crashing():
    runs = synthetic_runs(True, False)
    for run in runs:
        run["final"]["neural_collapse"]["nc3_self_duality_deviation"] = 1.0
    out = coordinate_audit(runs, "nc3_self_duality_deviation")
    assert np.isnan(out["pooled_within_condition"]["spearman"])
    assert np.isnan(out["pooled_within_condition"]["ci_95"][0])
