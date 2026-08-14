"""Retrieve and validate the official UCI Optical Digits archive."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from neural_geometry_lab.data import class_counts, load_official_splits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "optical-digits-uci.zip"
URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(URL, ARCHIVE)
    extracted: dict[str, dict[str, str | int]] = {}
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for target in ("optdigits.tra", "optdigits.tes"):
            matches = [name for name in bundle.namelist() if name.endswith(target)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {target}, found {matches}")
            destination = RAW / target
            destination.write_bytes(bundle.read(matches[0]))
            extracted[target] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
    train, test = load_official_splits(RAW)
    provenance = {
        "schema_version": "1.0.0",
        "source": "UCI Optical Recognition of Handwritten Digits",
        "doi": "10.24432/C50P49",
        "url": URL,
        "license": "CC BY 4.0",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "archive": {"bytes": ARCHIVE.stat().st_size, "sha256": sha256(ARCHIVE)},
        "files": extracted,
        "train": {"rows": int(train.labels.size), "class_counts": class_counts(train.labels)},
        "test": {"rows": int(test.labels.size), "class_counts": class_counts(test.labels)},
    }
    output = ROOT / "data" / "provenance-v1.0.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
