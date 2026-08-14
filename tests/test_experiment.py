from __future__ import annotations

import json

import numpy as np
import pytest

from neural_geometry_lab.data import DatasetSplit, construct_condition
from neural_geometry_lab.experiment import (
    _checkpoint,
    _initial_state,
    build_display_states,
    fit_logistic_reference,
    make_mlp,
    train_neural_run,
)


def split(rows_per_class: int = 3) -> DatasetSplit:
    labels = np.repeat(np.arange(10, dtype=np.int64), rows_per_class)
    features = np.zeros((labels.size, 64), dtype=np.float64)
    features[np.arange(labels.size), labels] = 1.0
    features[:, 20] = np.linspace(0.0, 0.2, labels.size)
    return DatasetSplit(features=features, labels=labels)


def test_initial_checkpoint_and_logistic_reference() -> None:
    train = split(3)
    test = split(2)
    condition = construct_condition(train, "clean")
    initial = _initial_state(condition, 1001)
    record, train_features, test_features, train_prediction, test_prediction = _checkpoint(
        initial, condition, test, 0
    )
    assert record["loss"] is None
    assert train_features.shape == (30, 9)
    assert test_features.shape == (20, 9)
    assert train_prediction.shape == (30,)
    assert test_prediction.shape == (20,)
    logistic = fit_logistic_reference(condition, test)
    assert logistic["test"]["accuracy"] == 1.0
    assert make_mlp(1001).hidden_layer_sizes == (64, 9)


def test_tiny_neural_run_is_serializable_and_keeps_checkpoints() -> None:
    train = split(13)
    test = split(2)
    condition = construct_condition(train, "clean")
    run = train_neural_run(condition, test, 1001, keep_display=True)
    assert len(run.checkpoints) == 13
    assert len(run.display_states) == 13
    assert run.summary["final"]["epoch"] == 600
    json.dumps(run.summary, allow_nan=False)


def test_display_builder_empty_input() -> None:
    train = split(3)
    condition = construct_condition(train, "clean")
    # The public caller never supplies no checkpoints; this verifies the failure is explicit.
    with pytest.raises(ValueError):
        build_display_states([], condition, split(2))
