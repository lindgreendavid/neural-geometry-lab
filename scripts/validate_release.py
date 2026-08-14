"""Validate committed v1.0 research and public-site artifacts without retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    results = load(ROOT / "reports" / "results-v1.0.json")
    site = load(ROOT / "site" / "data" / "study-v1.0.json")
    provenance = load(ROOT / "data" / "provenance-v1.0.json")
    assert results["schema_version"] == "1.0.0"
    assert results["product_version"] == "1.0.0"
    assert len(results["neural_runs"]) == 30  # type: ignore[arg-type]
    assert results["evaluation"] == site["summary"]
    assert results["dataset"] == site["dataset"]
    assert provenance["train"]["rows"] == 3823  # type: ignore[index]
    assert provenance["test"]["rows"] == 1797  # type: ignore[index]
    gates = results["evaluation"]["gates"]  # type: ignore[index]
    expected = {
        "clean_nc1_after_zero": (7, False),
        "clean_nc2_after_zero": (7, False),
        "clean_nc4_after_zero": (10, True),
        "imbalance_worsens_nc2": (10, True),
    }
    for name, (count, passed) in expected.items():
        assert gates[name] == {"count": count, "passed": passed}
    assert len(site["display_states"]) == 3  # type: ignore[arg-type]
    assert "Geometry theatre" in (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    print("v1.0 release artifacts are internally consistent")


if __name__ == "__main__":
    main()
