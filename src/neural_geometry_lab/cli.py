"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import class_counts, construct_condition, load_official_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the frozen Neural Geometry data source")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    train, test = load_official_splits(arguments.raw_dir)
    print(
        {
            "train_rows": int(train.labels.size),
            "test_rows": int(test.labels.size),
            "train_classes": class_counts(train.labels),
            "conditions": {
                name: int(construct_condition(train, name).observed_labels.size)
                for name in ("clean", "long_tail", "label_noise_20")
            },
        }
    )


if __name__ == "__main__":
    main()
