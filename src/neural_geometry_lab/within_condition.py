"""POST-HOC (not preregistered): collapse-vs-accuracy association within training conditions.

The frozen report states one descriptive Spearman association between final NC1 (or NC2) and test
accuracy across all 30 MLP runs. Those runs pool three training conditions whose accuracies differ
by more than ten points, so a pooled association can come entirely from the conditions rather than
from the collapse metric. This module reports, for every collapse coordinate (NC1-NC4):

* the pooled Spearman association (reproducing the frozen value);
* the Spearman association within each condition (10 algorithmic seeds each);
* the pooled *within-condition* association, i.e. ranks computed inside each condition;
* percentile bootstrap intervals that resample seeds within each condition.

Seeds are algorithmic replications, not samples of people or tasks, so the intervals describe
seed-to-seed variability only, consistent with the frozen report's decision not to report p-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from neural_geometry_lab.metrics import spearman

COORDINATES = {
    "nc1": "nc1_variability",
    "nc2": "nc2_simplex_deviation",
    "nc3": "nc3_self_duality_deviation",
    "nc4": "nc4_disagreement",
}
CONDITIONS = ("clean", "long_tail", "label_noise_20")
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 20_261_009


def _arrays(
    runs: list[dict[str, Any]], coordinate: str
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    out = {}
    for condition in CONDITIONS:
        rows = [run for run in runs if run["condition"] == condition]
        out[condition] = (
            np.array([float(r["final"]["neural_collapse"][coordinate]) for r in rows]),
            np.array([float(r["final"]["test"]["accuracy"]) for r in rows]),
        )
    return out


def _within_pooled(
    data: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> float:
    a = np.concatenate([rankdata(x) - (len(x) + 1) / 2 for x, _ in data.values()])
    b = np.concatenate([rankdata(y) - (len(y) + 1) / 2 for _, y in data.values()])
    return spearman(a, b)


def _interval(values: list[float]) -> list[float]:
    finite = np.array([v for v in values if not np.isnan(v)])
    if finite.size == 0:
        return [float("nan"), float("nan")]
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def coordinate_audit(runs: list[dict[str, Any]], coordinate: str) -> dict[str, Any]:
    data = _arrays(runs, coordinate)
    pooled_x = np.concatenate([x for x, _ in data.values()])
    pooled_y = np.concatenate([y for _, y in data.values()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_condition: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    pooled_boot: list[float] = []
    within_boot: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled = {}
        for condition, (x, y) in data.items():
            idx = rng.integers(0, len(x), len(x))
            resampled[condition] = (x[idx], y[idx])
            per_condition[condition].append(spearman(x[idx], y[idx]))
        pooled_boot.append(
            spearman(
                np.concatenate([x for x, _ in resampled.values()]),
                np.concatenate([y for _, y in resampled.values()]),
            )
        )
        within_boot.append(_within_pooled(resampled))
    return {
        "pooled": {"spearman": spearman(pooled_x, pooled_y), "ci_95": _interval(pooled_boot)},
        "pooled_within_condition": {
            "spearman": _within_pooled(data),
            "ci_95": _interval(within_boot),
        },
        "per_condition": {
            c: {
                "n": int(len(data[c][0])),
                "spearman": spearman(*data[c]),
                "ci_95": _interval(per_condition[c]),
            }
            for c in CONDITIONS
        },
    }


def build_within_condition(results: dict[str, Any]) -> dict[str, Any]:
    runs = results["neural_runs"]
    return {
        "schema_version": 1,
        "label": "POST-HOC (not preregistered): within-condition collapse-accuracy association",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "coordinates": {name: coordinate_audit(runs, field) for name, field in COORDINATES.items()},
        "frozen_pooled_spearman": results["evaluation"]["descriptive_correlations"],
    }
