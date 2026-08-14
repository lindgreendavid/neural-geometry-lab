from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from neural_geometry_lab.cli import main as cli_main
from neural_geometry_lab.data import (
    DatasetSplit,
    class_counts,
    construct_condition,
    load_digit_file,
    load_official_splits,
)


def synthetic_split(rows_per_class: int = 30) -> DatasetSplit:
    labels = np.repeat(np.arange(10, dtype=np.int64), rows_per_class)
    features = np.zeros((labels.size, 64), dtype=np.float64)
    features[np.arange(labels.size), labels] = 1.0
    return DatasetSplit(features=features, labels=labels)


def write_digit_file(path: Path, split: DatasetSplit) -> None:
    matrix = np.column_stack((np.rint(split.features * 16), split.labels))
    np.savetxt(path, matrix, fmt="%d", delimiter=",")


def test_digit_loader_and_official_pair(tmp_path: Path) -> None:
    split = synthetic_split(2)
    write_digit_file(tmp_path / "optdigits.tra", split)
    write_digit_file(tmp_path / "optdigits.tes", split)
    train, test = load_official_splits(tmp_path)
    assert train.features.shape == (20, 64)
    assert np.array_equal(train.labels, test.labels)
    assert class_counts(train.labels) == {str(index): 2 for index in range(10)}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda matrix: matrix.__setitem__((0, 0), 17), "Pixel values"),
        (lambda matrix: matrix.__setitem__((0, 64), 10), "Label outside"),
        (lambda matrix: matrix.__setitem__((0, 1), np.nan), "Non-finite"),
        (lambda matrix: matrix.__setitem__((0, 64), 1.5), "Non-integer"),
    ],
)
def test_loader_rejects_invalid_values(tmp_path: Path, mutator: object, message: str) -> None:
    split = synthetic_split(1)
    matrix = np.column_stack((split.features * 16, split.labels)).astype(np.float64)
    mutator(matrix)  # type: ignore[operator]
    path = tmp_path / "invalid.csv"
    np.savetxt(path, matrix, delimiter=",")
    with pytest.raises(ValueError, match=message):
        load_digit_file(path)


def test_loader_rejects_wrong_shape(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    np.savetxt(path, np.zeros((3, 3)), delimiter=",")
    with pytest.raises(ValueError, match="65 columns"):
        load_digit_file(path)


def test_frozen_conditions_are_deterministic() -> None:
    split = synthetic_split(100)
    clean = construct_condition(split, "clean")
    tail_a = construct_condition(split, "long_tail")
    tail_b = construct_condition(split, "long_tail")
    noisy = construct_condition(split, "label_noise_20")
    assert clean.observed_labels.size == 1000
    assert np.array_equal(tail_a.source_indices, tail_b.source_indices)
    assert class_counts(tail_a.observed_labels) == {
        "0": 100,
        "1": 77,
        "2": 59,
        "3": 46,
        "4": 35,
        "5": 27,
        "6": 25,
        "7": 25,
        "8": 25,
        "9": 25,
    }
    assert noisy.changed_labels == 200
    assert np.array_equal(noisy.true_labels, split.labels)
    with pytest.raises(ValueError, match="Unknown condition"):
        construct_condition(split, "unknown")


def test_command_line_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    split = synthetic_split(30)
    write_digit_file(tmp_path / "optdigits.tra", split)
    write_digit_file(tmp_path / "optdigits.tes", split)
    monkeypatch.setattr(sys, "argv", ["neural-geometry", "--raw-dir", str(tmp_path)])
    cli_main()
    output = capsys.readouterr().out
    assert "'train_rows': 300" in output
    assert "'label_noise_20': 300" in output
