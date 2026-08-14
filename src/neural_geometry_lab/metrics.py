"""Prediction and neural-collapse endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, log_loss

from .data import FloatArray, IntArray


def relu(values: FloatArray) -> FloatArray:
    return np.maximum(values, 0.0).astype(np.float64)


def forward_features(model: Any, features: FloatArray) -> FloatArray:
    """Return the learned nine-dimensional penultimate activations."""
    activation = features
    for weights, intercept in zip(model.coefs_[:-1], model.intercepts_[:-1], strict=True):
        activation = relu(activation @ weights + intercept)
    return activation.astype(np.float64)


def expected_calibration_error(
    probabilities: FloatArray, labels: IntArray, *, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(mask.sum())
        if count:
            accuracy = float(np.mean(prediction[mask] == labels[mask]))
            mean_confidence = float(confidence[mask].mean())
            value += count / total * abs(accuracy - mean_confidence)
    return float(value)


def prediction_metrics(probabilities: FloatArray, labels: IntArray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(10, dtype=np.float64)[labels]
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(10))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_15": expected_calibration_error(probabilities, labels),
    }


def class_means(features: FloatArray, labels: IntArray) -> FloatArray:
    means = []
    for class_id in range(10):
        subset = features[labels == class_id]
        if subset.size == 0:
            raise ValueError(f"Class {class_id} is absent")
        means.append(subset.mean(axis=0))
    return np.asarray(np.stack(means), dtype=np.float64)


def nearest_center_prediction(features: FloatArray, means: FloatArray) -> IntArray:
    distances = np.sum((features[:, None, :] - means[None, :, :]) ** 2, axis=2)
    return np.asarray(distances.argmin(axis=1), dtype=np.int64)


def neural_collapse_metrics(
    features: FloatArray,
    labels: IntArray,
    classifier_weights: FloatArray,
    network_prediction: IntArray,
) -> dict[str, float | list[list[float]]]:
    means = class_means(features, labels)
    global_mean = features.mean(axis=0)
    centered_means = means - global_mean
    within_trace = 0.0
    for class_id in range(10):
        centered = features[labels == class_id] - means[class_id]
        within_trace += float(np.sum(centered * centered))
    within_trace /= features.shape[0]
    between_trace = float(np.mean(np.sum(centered_means * centered_means, axis=1)))
    nc1 = within_trace / between_trace if between_trace > 0 else float("inf")

    mean_norms = np.linalg.norm(centered_means, axis=1, keepdims=True)
    normalized_means = centered_means / np.maximum(mean_norms, 1e-12)
    gram = normalized_means @ normalized_means.T
    target = np.full((10, 10), -1 / 9, dtype=np.float64)
    np.fill_diagonal(target, 1.0)
    nc2 = float(np.linalg.norm(gram - target) / np.linalg.norm(target))

    weight_vectors = classifier_weights.T
    centered_weights = weight_vectors - weight_vectors.mean(axis=0)
    weight_norms = np.linalg.norm(centered_weights, axis=1)
    class_norms = np.linalg.norm(centered_means, axis=1)
    cosine = np.sum(centered_weights * centered_means, axis=1) / np.maximum(
        weight_norms * class_norms, 1e-12
    )
    nc3 = float(1.0 - np.mean(cosine))

    nearest = nearest_center_prediction(features, means)
    nc4 = float(np.mean(nearest != network_prediction))
    return {
        "nc1_variability": float(nc1),
        "nc2_simplex_deviation": nc2,
        "nc3_self_duality_deviation": nc3,
        "nc4_disagreement": nc4,
        "class_angle_gram": np.round(gram, 8).tolist(),
    }


def spearman(values_a: NDArray[np.float64], values_b: NDArray[np.float64]) -> float:
    from scipy.stats import rankdata

    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])
