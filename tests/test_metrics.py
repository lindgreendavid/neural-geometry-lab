from __future__ import annotations

import numpy as np
import pytest

from neural_geometry_lab.metrics import (
    class_means,
    expected_calibration_error,
    forward_features,
    nearest_center_prediction,
    neural_collapse_metrics,
    prediction_metrics,
    relu,
    spearman,
)


class TinyModel:
    coefs_ = [np.eye(2), np.eye(2)]
    intercepts_ = [np.zeros(2), np.zeros(2)]


def test_prediction_metrics_and_calibration() -> None:
    labels = np.arange(10, dtype=np.int64)
    probabilities = np.eye(10, dtype=np.float64)
    metrics = prediction_metrics(probabilities, labels)
    assert metrics == pytest.approx(
        {"accuracy": 1.0, "macro_f1": 1.0, "log_loss": 0.0, "brier": 0.0, "ece_15": 0.0}
    )
    uncertain = np.full((10, 10), 0.1, dtype=np.float64)
    assert expected_calibration_error(uncertain, labels, bins=2) == pytest.approx(0.0)


def test_features_means_and_nearest_centres() -> None:
    features = np.asarray([[-1.0, 2.0], [3.0, -4.0]])
    assert np.array_equal(relu(features), np.asarray([[0.0, 2.0], [3.0, 0.0]]))
    assert np.array_equal(forward_features(TinyModel(), features), relu(features))
    repeated = np.repeat(np.eye(10, dtype=np.float64), 2, axis=0)
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    means = class_means(repeated, labels)
    assert np.array_equal(nearest_center_prediction(repeated, means), labels)
    with pytest.raises(ValueError, match="Class 9 is absent"):
        class_means(repeated[:-2], labels[:-2])


def test_neural_collapse_metrics_on_exact_simplex() -> None:
    identity = np.eye(10, dtype=np.float64)
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    features = np.repeat(identity, 2, axis=0)
    result = neural_collapse_metrics(features, labels, identity, labels)
    assert result["nc1_variability"] == pytest.approx(0.0)
    assert result["nc2_simplex_deviation"] == pytest.approx(0.0, abs=1e-12)
    assert result["nc3_self_duality_deviation"] == pytest.approx(0.0, abs=1e-12)
    assert result["nc4_disagreement"] == pytest.approx(0.0)
    assert len(result["class_angle_gram"]) == 10  # type: ignore[arg-type]


def test_rank_association_and_constant_boundary() -> None:
    ascending = np.asarray([1.0, 2.0, 3.0])
    descending = ascending[::-1]
    assert spearman(ascending, descending) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(3), descending))
