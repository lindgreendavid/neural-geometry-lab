"""Frozen model training and checkpoint extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .data import DatasetSplit, FloatArray, IntArray, TrainingCondition
from .metrics import (
    class_means,
    forward_features,
    nearest_center_prediction,
    neural_collapse_metrics,
    prediction_metrics,
)

CHECKPOINTS = (0, 1, 2, 5, 10, 20, 40, 80, 120, 200, 300, 450, 600)
SEEDS = tuple(range(1001, 1011))
CONDITIONS = ("clean", "long_tail", "label_noise_20")


@dataclass
class NeuralRun:
    summary: dict[str, Any]
    checkpoints: list[dict[str, Any]]
    display_states: list[dict[str, Any]]


def make_mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 9),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        warm_start=True,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=1000,
    )


def _initial_state(condition: TrainingCondition, seed: int) -> MLPClassifier:
    """Initialize the pinned sklearn model without an optimization step."""
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils import check_random_state

    model = make_mlp(seed)
    model._label_binarizer = LabelBinarizer()
    encoded = model._label_binarizer.fit_transform(condition.observed_labels)
    if encoded.shape[1] == 1:
        encoded = np.hstack([1 - encoded, encoded])
    model.classes_ = model._label_binarizer.classes_
    model.n_outputs_ = encoded.shape[1]
    model._random_state = check_random_state(seed)
    model._initialize(encoded, [64, 64, 9, 10], np.dtype(np.float64))
    model.n_features_in_ = 64
    return model


def _checkpoint_unchecked(
    model: MLPClassifier,
    condition: TrainingCondition,
    test: DatasetSplit,
    epoch: int,
) -> tuple[dict[str, Any], FloatArray, FloatArray, IntArray, IntArray]:
    train_probability = model.predict_proba(condition.features).astype(np.float64)
    test_probability = model.predict_proba(test.features).astype(np.float64)
    train_prediction = train_probability.argmax(axis=1).astype(np.int64)
    test_prediction = test_probability.argmax(axis=1).astype(np.int64)
    train_features = forward_features(model, condition.features)
    test_features = forward_features(model, test.features)
    train_nc = neural_collapse_metrics(
        train_features,
        condition.observed_labels,
        model.coefs_[-1].astype(np.float64),
        train_prediction,
    )
    means = class_means(train_features, condition.observed_labels)
    nearest_test = nearest_center_prediction(test_features, means)
    raw_loss = float(getattr(model, "loss_", np.nan))
    record: dict[str, Any] = {
        "epoch": epoch,
        "train": prediction_metrics(train_probability, condition.observed_labels),
        "train_true_labels": prediction_metrics(train_probability, condition.true_labels),
        "test": prediction_metrics(test_probability, test.labels),
        "neural_collapse": train_nc,
        "test_nc4_disagreement": float(np.mean(nearest_test != test_prediction)),
        "loss": raw_loss if np.isfinite(raw_loss) else None,
    }
    return record, train_features, test_features, train_prediction, test_prediction


def _checkpoint(
    model: MLPClassifier,
    condition: TrainingCondition,
    test: DatasetSplit,
    epoch: int,
) -> tuple[dict[str, Any], FloatArray, FloatArray, IntArray, IntArray]:
    """Evaluate a checkpoint and reject any non-finite numerical state."""
    with np.errstate(all="ignore"):
        result = _checkpoint_unchecked(model, condition, test, epoch)
    arrays = [*model.coefs_, *model.intercepts_, result[1], result[2]]
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError(f"Non-finite model state at epoch {epoch}")
    return result


def _display_indices(labels: IntArray, per_class: int) -> IntArray:
    pieces = [np.flatnonzero(labels == class_id)[:per_class] for class_id in range(10)]
    return np.asarray(np.concatenate(pieces), dtype=np.int64)


def train_neural_run(
    condition: TrainingCondition,
    test: DatasetSplit,
    seed: int,
    *,
    keep_display: bool,
) -> NeuralRun:
    model = make_mlp(seed)
    initial = _initial_state(condition, seed)
    checkpoints: list[dict[str, Any]] = []
    raw_states: list[tuple[int, dict[str, Any], FloatArray, FloatArray, IntArray, IntArray]] = []
    (
        initial_record,
        initial_train_features,
        initial_test_features,
        initial_train_prediction,
        initial_test_prediction,
    ) = _checkpoint(initial, condition, test, 0)
    checkpoints.append(initial_record)
    if keep_display:
        raw_states.append(
            (
                0,
                initial_record,
                initial_train_features,
                initial_test_features,
                initial_train_prediction,
                initial_test_prediction,
            )
        )

    first_zero_epoch: int | None = None
    first_zero_record: dict[str, Any] | None = None
    final_record = initial_record
    for epoch in range(1, 601):
        with np.errstate(all="ignore"):
            if epoch == 1:
                model.partial_fit(
                    condition.features, condition.observed_labels, classes=np.arange(10)
                )
            else:
                model.partial_fit(condition.features, condition.observed_labels)
            train_prediction = model.predict(condition.features)
        if first_zero_epoch is None and bool(np.all(train_prediction == condition.observed_labels)):
            first_zero_epoch = epoch
            first_zero_record, _, _, _, _ = _checkpoint(model, condition, test, epoch)
        if epoch in CHECKPOINTS:
            record, train_features, test_features, train_prediction, test_prediction = _checkpoint(
                model, condition, test, epoch
            )
            checkpoints.append(record)
            final_record = record
            if keep_display:
                raw_states.append(
                    (
                        epoch,
                        record,
                        train_features,
                        test_features,
                        train_prediction,
                        test_prediction,
                    )
                )

    display_states = build_display_states(raw_states, condition, test) if keep_display else []
    summary = {
        "condition": condition.name,
        "seed": seed,
        "first_zero_epoch": first_zero_epoch,
        "first_zero": first_zero_record,
        "final": final_record,
    }
    return NeuralRun(summary=summary, checkpoints=checkpoints, display_states=display_states)


def build_display_states(
    raw_states: list[tuple[int, dict[str, Any], FloatArray, FloatArray, IntArray, IntArray]],
    condition: TrainingCondition,
    test: DatasetSplit,
) -> list[dict[str, Any]]:
    from sklearn.decomposition import PCA

    train_indices = _display_indices(condition.observed_labels, 18)
    test_indices = _display_indices(test.labels, 8)
    fitting = []
    for _, _, train_features, test_features, _, _ in raw_states:
        fitting.extend([train_features[train_indices], test_features[test_indices]])
    projector = PCA(n_components=2, svd_solver="full").fit(np.vstack(fitting))
    states: list[dict[str, Any]] = []
    for (
        epoch,
        record,
        train_features,
        test_features,
        train_predictions,
        test_predictions,
    ) in raw_states:
        train_xy = projector.transform(train_features[train_indices])
        test_xy = projector.transform(test_features[test_indices])
        means = class_means(train_features, condition.observed_labels)
        mean_xy = projector.transform(means)
        train_prediction = train_predictions[train_indices]
        test_prediction = test_predictions[test_indices]
        points = []
        for row, source_index, prediction in zip(
            train_xy, train_indices, train_prediction, strict=True
        ):
            points.append(
                {
                    "x": round(float(row[0]), 6),
                    "y": round(float(row[1]), 6),
                    "label": int(condition.observed_labels[source_index]),
                    "true_label": int(condition.true_labels[source_index]),
                    "prediction": int(prediction),
                    "split": "train",
                }
            )
        for row, source_index, prediction in zip(
            test_xy, test_indices, test_prediction, strict=True
        ):
            points.append(
                {
                    "x": round(float(row[0]), 6),
                    "y": round(float(row[1]), 6),
                    "label": int(test.labels[source_index]),
                    "true_label": int(test.labels[source_index]),
                    "prediction": int(prediction),
                    "split": "test",
                }
            )
        states.append(
            {
                "epoch": epoch,
                "points": points,
                "centres": [
                    {"class": class_id, "x": round(float(row[0]), 6), "y": round(float(row[1]), 6)}
                    for class_id, row in enumerate(mean_xy)
                ],
                "metrics": record,
            }
        )
    return states


def fit_logistic_reference(condition: TrainingCondition, test: DatasetSplit) -> dict[str, Any]:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    with np.errstate(all="ignore"):
        model.fit(condition.features, condition.observed_labels)
        train_probability = model.predict_proba(condition.features).astype(np.float64)
        test_probability = model.predict_proba(test.features).astype(np.float64)
    if not all(
        np.isfinite(array).all()
        for array in (model.coef_, model.intercept_, train_probability, test_probability)
    ):
        raise FloatingPointError(f"Non-finite logistic state for {condition.name}")
    return {
        "condition": condition.name,
        "iterations": [int(value) for value in model.n_iter_],
        "train": prediction_metrics(train_probability, condition.observed_labels),
        "train_true_labels": prediction_metrics(train_probability, condition.true_labels),
        "test": prediction_metrics(test_probability, test.labels),
    }
