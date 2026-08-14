# Neural Geometry Lab

[![CI](https://github.com/lindgreendavid/neural-geometry-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/lindgreendavid/neural-geometry-lab/actions/workflows/ci.yml)
[![CodeQL](https://github.com/lindgreendavid/neural-geometry-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/lindgreendavid/neural-geometry-lab/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/lindgreendavid/neural-geometry-lab)](https://github.com/lindgreendavid/neural-geometry-lab/releases/tag/v1.0.0)
[![Licence: MIT](https://img.shields.io/badge/License-MIT-11131b.svg)](LICENSE)

> After a small neural classifier reaches zero training error, does its hidden representation
> continue toward classical neural-collapse geometry—and does stronger collapse consistently
> accompany better generalization to unseen writers?

Neural Geometry Lab is a frozen, reproducible comparison of multinomial logistic regression and a
small ReLU network on the official writer-disjoint UCI Optical Digits split. It measures NC1–NC4 and
stress-tests their interpretation under deterministic class imbalance and 20% symmetric label noise.

## What this contributes

This project contributes a protocol-frozen, seed-resolved test of whether four neural-collapse
coordinates continue after interpolation and consistently accompany writer-disjoint generalization.
The public theatre replays saved checkpoints while keeping its two-dimensional PCA view separate
from the full nine-dimensional endpoints. It does **not** establish collapse as causal, necessary,
sufficient, or representative of larger architectures and datasets.

**[Open the interactive lab](https://lindgreendavid.github.io/neural-geometry-lab/)** ·
**[Read the research report](reports/research-report-v1.0.md)** ·
**[Inspect the frozen protocol](docs/protocol-v1.0.md)**

## v1.0 result

The preregistered result is a partial confirmation, not a universal-collapse claim:

- clean NC4 remained no worse after interpolation in **10/10** seeds;
- deterministic imbalance worsened final NC2 in **10/10** paired seeds;
- continued clean NC1 and NC2 improvement occurred in **7/10** seeds each, below the frozen 8/10
  gates;
- clean MLP median held-out accuracy was **96.27%** (range 95.88–96.94%);
- across all 30 MLP runs, descriptive Spearman associations with accuracy were **−0.518** for NC1
  and **−0.026** for NC2.

The noisy-label condition generalized much worse while its median NC2 was slightly lower than clean.
That does not refute neural collapse. It shows why one geometry coordinate is not a generalization
certificate.

## Interactive evidence

The Geometry Theatre animates actual saved checkpoints from seed 1001 in each condition. It combines:

- a fixed-basis PCA projection of sampled training and held-out representations;
- a full nine-dimensional class-angle wheel;
- live NC1–NC4 and held-out accuracy trajectories;
- visible prediction errors, training/test controls, and reduced-motion support.

PCA powers only the explorable display. Every reported neural-collapse endpoint is computed in the
full nine-dimensional hidden representation.

## Reproduce

Requires Python 3.10–3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/fetch_data.py
python scripts/run_study.py
python scripts/validate_release.py
```

The fetcher downloads only the official UCI archive and verifies frozen SHA-256 identities. The study
regenerates the result register, trajectories, and public-site data. Raw UCI files remain uncommitted.

## Evidence map

| Artifact | Purpose |
| --- | --- |
| [`docs/protocol-v1.0.md`](docs/protocol-v1.0.md) | Frozen question, design, endpoints, and gates |
| [`docs/source-audit.md`](docs/source-audit.md) | Primary literature and exact dataset identity |
| [`reports/results-v1.0.json`](reports/results-v1.0.json) | Complete machine-readable evaluation |
| [`reports/research-report-v1.0.md`](reports/research-report-v1.0.md) | Methods, results, interpretation, and limits |
| [`reports/trajectory-v1.0.json`](reports/trajectory-v1.0.json) | Seed-1001 checkpoint trajectories |
| [`docs/v1-release-audit.md`](docs/v1-release-audit.md) | Reproduction actions, deviations, and boundaries |
| [`data/provenance-v1.0.json`](data/provenance-v1.0.json) | Checksums, counts, retrieval time, and licence |

## Scientific boundary

This is one compact architecture, optimizer, dataset, and writer split. The ten seeds quantify
algorithmic sensitivity, not population uncertainty. The synthetic label noise is not a model of
real annotation processes. The work does not make claims about brains, foundation models, causal
representation learning, or deployed systems.

## Governance and licence

Scientific changes follow the frozen-artifact rules in [CONTRIBUTING.md](CONTRIBUTING.md). Security
reports use private GitHub advisories as described in [SECURITY.md](SECURITY.md). Original software
and documentation are MIT licensed; UCI data retain CC BY 4.0 attribution.
