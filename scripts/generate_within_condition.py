"""Generate the POST-HOC within-condition association audit from the frozen results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_geometry_lab.within_condition import build_within_condition

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results-v1.0.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "post-release-within-condition.json"
    )
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(build_within_condition(results), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
