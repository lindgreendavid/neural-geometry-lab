# Neural Geometry Lab v1.0 research report

## Abstract

Neural collapse describes four related geometric regularities often observed late in supervised
classification. We tested whether those regularities continue to strengthen after interpolation and
whether their strength tracks held-out performance in a deliberately small, inspectable setting.
A two-hidden-layer multilayer perceptron (MLP) and an L2-regularized multinomial logistic regression
reference were evaluated on the official writer-disjoint UCI Optical Digits split under clean,
deterministically imbalanced, and 20% symmetric label-noise training conditions. The frozen MLP
experiment comprised ten algorithmic seeds per condition and 600 epochs per seed. Two of four
prespecified directional gates passed. Clean-seed NC4 disagreement did not increase after the first
zero-error epoch in 10/10 seeds, and imbalance worsened final NC2 simplex deviation in 10/10 paired
seeds. Continued improvement in clean NC1 and NC2 occurred in only 7/10 seeds each, below the
prespecified 8/10 threshold. Clean MLP held-out accuracy had a median of 96.27% (range 95.88–96.94%).
Across all 30 algorithmic runs, final NC1 had a descriptive Spearman association of −0.518 with test
accuracy, while final NC2 was nearly unassociated (−0.026). In this bounded experiment, some collapse
properties were robust, but proximity to an equiangular simplex was not a sufficient summary of
writer-disjoint generalization.

## Question and status

The confirmatory question was frozen before data retrieval and model fitting:

> After a small neural classifier reaches zero training error, does its hidden representation
> continue toward classical neural-collapse geometry, and does stronger collapse consistently
> accompany better generalization to unseen writers?

This is a controlled computational replication and stress test, not a claim about all architectures,
datasets, optimization regimes, or biological learning. The protocol is `docs/protocol-v1.0.md`; the
complete machine-readable result is `reports/results-v1.0.json`.

## Data and conditions

We used the official UCI Optical Recognition of Handwritten Digits files, normalized only by dividing
the 64 integer pixel-block counts by 16. The provider's fixed split contains 3,823 training examples
and 1,797 test examples from disjoint writer sets. Dataset identity, checksums, counts, licence, and
retrieval time are recorded in `data/provenance-v1.0.json` and `docs/source-audit.md`.

The three training conditions were:

1. **Clean:** all official training observations and labels.
2. **Long tail:** class `c` retained `floor(n_c × 10^(−c/9))` observations, with a minimum of 25,
   producing 1,559 rows with class counts from 376 (digit 0) to 38 (digit 9).
3. **20% label noise:** exactly `floor(0.20 × n_c)` labels per class were changed to another class,
   producing 761 changed labels. Selection and replacement used construction seed 20260814.

No test observation influenced condition construction, training, stopping, or model selection.

## Models and outcomes

The reference was multinomial logistic regression with L2 regularization (`C=1`). The neural model was
a `64 → 64 → 9 → 10` ReLU MLP trained with Adam for exactly 600 epochs, batch size 128, learning rate
0.001, and weight penalty 0.0001. Seeds 1001–1010 were algorithmic replications, not sampled people or
datasets.

We computed the four classical diagnostics at frozen checkpoints:

- **NC1:** within-class variability, `trace(S_W) / trace(S_B)`; lower is more collapsed.
- **NC2:** relative Frobenius deviation of normalized centred class means from the ten-class simplex
  equiangular tight-frame Gram matrix; lower is closer to the ideal simplex.
- **NC3:** one minus the mean cosine alignment between classifier weights and centred class means;
  lower is more self-dual.
- **NC4:** disagreement between the learned classifier and nearest-class-centre decisions; lower is
  greater decision agreement.

Accuracy, macro-F1, multiclass log loss, Brier score, and 15-bin expected calibration error were also
recorded. The interactive two-dimensional view uses one PCA basis fitted across the displayed
checkpoints within each condition. It is a projection for inspection; every reported NC metric is
computed in the full nine-dimensional hidden space.

## Confirmatory results

| Prespecified gate | Result | Decision |
| --- | ---: | --- |
| Final NC1 below first-zero NC1 in at least 8/10 clean seeds | 7/10 | Not passed |
| Final NC2 below first-zero NC2 in at least 8/10 clean seeds | 7/10 | Not passed |
| Final NC4 no greater than first-zero NC4 in at least 8/10 clean seeds | 10/10 | Passed |
| Long-tail final NC2 worse than clean in at least 8/10 paired seeds | 10/10 | Passed |

All ten clean and all ten long-tail runs reached zero observed-label training error. Their first
zero-error epochs ranged from 117 to 216 in the clean condition. No noisy-label run reached zero error
within 600 epochs.

| MLP condition | Median test accuracy | Seed range | Median NC1 | Median NC2 | Zero-error seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 96.27% | 95.88–96.94% | 0.2594 | 1.0077 | 10/10 |
| Long tail | 92.29% | 91.04–93.66% | 0.2295 | 1.0771 | 10/10 |
| 20% label noise | 81.64% | 79.41–83.58% | 1.2532 | 0.9868 | 0/10 |

The logistic reference achieved 94.77%, 91.88%, and 93.60% test accuracy in the clean, long-tail, and
noisy-label conditions respectively. The noisy-label comparison is particularly instructive: the
fixed-capacity logistic model resisted label memorization better than the 600-epoch MLP, while the MLP
never interpolated the corrupted labels. This is a model-and-training-regime result, not a universal
claim that linear models are more noise-robust.

Across all 30 MLP runs, the descriptive Spearman correlations between test accuracy and final NC1 and
NC2 were −0.518 and −0.026. No population-level p-values are reported because algorithmic seeds are
not independent samples from a human or task population.

## Interpretation

The evidence supports a qualified conclusion. Decision self-duality (NC4) remained stable after
interpolation, and deterministic class imbalance reliably displaced the class means farther from the
balanced simplex target. Those observations agree with the view that imbalance changes terminal
geometry. However, continued NC1 and NC2 movement was not seed-universal under the prespecified gate,
and final NC2 alone did not track held-out accuracy across stress conditions.

The label-noise condition makes the boundary visible. Its median NC2 was numerically slightly lower
than the clean median even while generalization deteriorated sharply and NC1 increased almost
fivefold. Thus a representation can be closer to the ideal class-mean Gram matrix by one diagnostic
without being globally more collapsed or more useful out of sample. This does not refute neural
collapse; it rejects the stronger shortcut that any single collapse coordinate is a general-purpose
certificate of generalization.

## Relationship to prior work

Papyan, Han, and Donoho established the NC1–NC4 terminology and documented terminal-phase collapse in
standard deep networks. Dang and colleagues showed theoretically and empirically that class imbalance
changes the geometry under an unconstrained ReLU-features model. Wu and Mondelli developed conditions
linking collapse and generalization in a mean-field regime. Han and colleagues subsequently argued,
through grokking experiments, that neural collapse need not be necessary for generalization. Our small
experiment is compatible with the shared boundary implied by this literature: collapse is a
multi-coordinate, regime-dependent phenomenon, and a geometric diagnostic should not be elevated into
an architecture-independent causal explanation.

## Limitations

- One compact tabular image dataset, one official writer split, one MLP architecture, and one optimizer
  were studied.
- The ten seeds quantify algorithmic sensitivity under a fixed dataset; they do not establish external
  statistical generality.
- NC2 uses the balanced ten-class simplex as its reference even under imbalance, deliberately testing
  deviation from classical balanced collapse rather than fitting an imbalance-specific target.
- The noisy condition changes training labels only and represents symmetric synthetic corruption, not
  realistic annotator disagreement or distribution shift.
- PCA can distort distance and angle. It powers only the animation; all metrics use the full hidden
  representation.
- The 8/10 gates are prespecified directional robustness criteria, not calibrated frequentist tests.

## Reproduction

Retrieve and verify the official archive with `python scripts/fetch_data.py`, then execute
`python scripts/run_study.py`. The run regenerates the full result register, the checkpoint trajectory,
and the data used by the public site. Reproduction should match exactly under the pinned environment;
any platform-level floating-point difference must preserve the reported decisions and is to be logged
in the release audit rather than silently normalized.

## References

1. Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine
   Learning Repository. <https://doi.org/10.24432/C50P49>
2. Papyan, V., Han, X. Y. & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal
   phase of deep learning training*. PNAS 117, 24652–24663.
   <https://doi.org/10.1073/pnas.2015509117>
3. Dang, H., Tran Huu, T., Nguyen, T. M. & Ho, N. (2024). *Neural Collapse for Cross-entropy
   Class-Imbalanced Learning with Unconstrained ReLU Features Model*. ICML 2024.
   <https://proceedings.mlr.press/v235/dang24a.html>
4. Wu, D. & Mondelli, M. (2025). *Neural Collapse Beyond the Unconstrained Features Model: Landscape,
   Dynamics, and Generalization in the Mean-Field Regime*. ICML 2025.
   <https://proceedings.mlr.press/v267/wu25u.html>
5. Han, T. et al. (2025). *Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via
   Grokking*. NeurIPS 2025. <https://openreview.net/forum?id=lbtOctHDQ3>
