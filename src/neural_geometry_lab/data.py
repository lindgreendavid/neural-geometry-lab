"""Strict loading and frozen condition construction for Optical Digits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DatasetSplit:
    features: FloatArray
    labels: IntArray


@dataclass(frozen=True)
class TrainingCondition:
    name: str
    features: FloatArray
    observed_labels: IntArray
    true_labels: IntArray
    source_indices: IntArray
    changed_labels: int


def load_digit_file(path: Path) -> DatasetSplit:
    """Load one official 64-feature comma-separated digit file."""
    matrix = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 65:
        raise ValueError(f"Expected 65 columns in {path}, found {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite value in {path}")
    pixels = matrix[:, :64]
    raw_labels = matrix[:, 64]
    if (pixels < 0).any() or (pixels > 16).any() or not np.equal(pixels, np.floor(pixels)).all():
        raise ValueError(f"Pixel values outside the documented integer range in {path}")
    if not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError(f"Non-integer label in {path}")
    labels = raw_labels.astype(np.int64)
    if (labels < 0).any() or (labels > 9).any():
        raise ValueError(f"Label outside 0-9 in {path}")
    return DatasetSplit(features=(pixels / 16.0).astype(np.float64), labels=labels)


def load_official_splits(raw_dir: Path) -> tuple[DatasetSplit, DatasetSplit]:
    return (
        load_digit_file(raw_dir / "optdigits.tra"),
        load_digit_file(raw_dir / "optdigits.tes"),
    )


def construct_condition(
    train: DatasetSplit, name: str, *, construction_seed: int = 20260814
) -> TrainingCondition:
    """Apply one prespecified training-only condition."""
    rng = np.random.default_rng(construction_seed)
    all_indices = np.arange(train.labels.size, dtype=np.int64)
    selected: IntArray
    if name == "clean":
        selected = all_indices
        observed = train.labels.copy()
    elif name == "long_tail":
        pieces: list[IntArray] = []
        for class_id in range(10):
            class_indices = np.flatnonzero(train.labels == class_id).astype(np.int64)
            count = max(25, int(np.floor(class_indices.size * 10 ** (-class_id / 9))))
            pieces.append(np.sort(rng.choice(class_indices, size=count, replace=False)))
        selected = np.sort(np.concatenate(pieces)).astype(np.int64)
        observed = train.labels[selected].copy()
    elif name == "label_noise_20":
        selected = all_indices
        observed = train.labels.copy()
        for class_id in range(10):
            class_indices = np.flatnonzero(train.labels == class_id)
            count = int(np.floor(0.20 * class_indices.size))
            changed = rng.choice(class_indices, size=count, replace=False)
            offsets = rng.integers(1, 10, size=count)
            observed[changed] = (class_id + offsets) % 10
    else:
        raise ValueError(f"Unknown condition: {name}")
    true_labels = train.labels[selected].copy()
    changed_labels = int(np.count_nonzero(observed != true_labels))
    return TrainingCondition(
        name=name,
        features=train.features[selected].copy(),
        observed_labels=observed,
        true_labels=true_labels,
        source_indices=selected,
        changed_labels=changed_labels,
    )


def class_counts(labels: IntArray) -> dict[str, int]:
    return {str(class_id): int(np.count_nonzero(labels == class_id)) for class_id in range(10)}
