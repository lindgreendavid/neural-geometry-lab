# Frozen research protocol v1.0

Status: frozen before retrieval of the digit records and before any model result was inspected.

## Question

After a small neural classifier reaches zero training error, do its penultimate-layer features
continue toward the four classical neural-collapse properties, and does stronger collapse on the
training sample consistently accompany better generalization to the official writer-disjoint test
sample?

This is a controlled reproduction and boundary study. It is not a claim that neural collapse causes
generalization or that this small network represents modern foundation models.

## Evidence

The empirical source is UCI Optical Recognition of Handwritten Digits, DOI
`10.24432/C50P49`, licensed CC BY 4.0. The source contains 5,620 normalized 8 × 8 digit records.
The provider constructed the training source from 30 writers and the test source from a different
13 writers. The official `optdigits.tra` and `optdigits.tes` files define the only final split.

The conceptual endpoints follow Papyan, Han and Donoho (2020). The imbalance boundary follows the
distinction in Dang et al. (2024) between within-class collapse and symmetric simplex geometry.
Recent work that questions a necessary relationship between collapse and generalization motivates
the descriptive discordance analysis; it does not predetermine its result.

## Preprocessing

- Parse all 64 integer pixels and one integer class label from the two official files.
- Reject malformed rows, non-finite values, pixels outside 0–16, and labels outside 0–9.
- Divide pixels by 16. No centering, augmentation, feature selection, or test-dependent transform.
- Never use the official test source for tuning, early stopping, architecture choice, or condition
  construction.

## Models

### Linear reference

Multinomial logistic regression with L2 regularization, `C = 1`, an intercept, and a converged
quasi-Newton solver. It supplies predictive and calibration references only. It has no learned
penultimate representation, so NC1–NC4 are not assigned to it.

### Neural classifier

A fully connected network `64 → 64 → 9 → 10` with ReLU hidden activations and a softmax output.
The nine-dimensional penultimate layer permits, but does not force, the ideal geometry of ten
centred class means. Training uses cross-entropy, Adam, batch size 128, learning rate 0.001, L2
coefficient 0.0001, deterministic seed-specific shuffling, and 600 complete epochs. There is no
early stopping. Seeds are `1001` through `1010`.

Checkpoint epochs are `0, 1, 2, 5, 10, 20, 40, 80, 120, 200, 300, 450, 600`. Metrics are recorded
every epoch so the first zero-training-error epoch can be identified exactly. Stored browser
features are restricted to the checkpoints and a fixed stratified display sample.

## Prespecified training conditions

1. `clean`: the complete official training source.
2. `long_tail`: for class `c`, retain a deterministic subset of
   `floor(n_c × 10^(−c/9))` records, with a floor of 25. The condition intentionally assigns the
   largest class to digit 0 and the smallest to digit 9; it estimates no natural prevalence.
3. `label_noise_20`: within every class, deterministically select `floor(0.20 × n_c)` training
   records and replace each observed label by a uniformly selected different label. True labels are
   retained only for diagnostic evaluation.

Condition construction uses seed `20260814` and is identical across model seeds.

## Endpoints

Let `h_i` be a penultimate feature, `μ_c` its observed-label class mean, `μ_G` the global mean,
`S_W` the pooled within-class covariance, and `S_B` the covariance of the ten class means.

- **NC1 variability ratio:** `trace(S_W) / trace(S_B)`. Lower is more collapsed.
- **NC2 simplex deviation:** relative Frobenius distance between the Gram matrix of normalized,
  centred class means and the ten-class simplex target with diagonal 1 and off-diagonal `−1/9`.
  Lower is closer to the balanced simplex geometry.
- **NC3 self-duality deviation:** one minus the mean cosine alignment between centred classifier
  weight vectors and centred class means. Lower is more aligned.
- **NC4 disagreement:** fraction for which the network prediction differs from the nearest training
  class mean in penultimate Euclidean distance. Train and official test values are separate.
- **Prediction:** accuracy, macro-F1, multiclass log loss, Brier score, and 15-bin equal-width ECE.

All NC metrics use the full nine-dimensional features. Browser projections never replace an
endpoint. For noisy training, training NC metrics use the labels actually optimized; test metrics
use the unmodified true labels.

## Frozen evaluation gates

The clean-condition reproduction passes each directional gate when at least 8 of 10 seeds satisfy:

1. final NC1 is below NC1 at the first zero-training-error epoch;
2. final NC2 is below NC2 at the first zero-training-error epoch;
3. final training NC4 disagreement is no greater than at that epoch.

The imbalance boundary is supported when at least 8 paired seeds have worse final NC2 under
`long_tail` than under `clean`. Label-noise effects and all relationships between NC metrics and
official test accuracy are descriptive, with seed-level values, medians, ranges, and Spearman
correlations reported. Seeds are algorithmic repeats on one fixed dataset, not independent human
samples; no population p-value is attached to them.

## Projection and interaction

The animated two-dimensional representation uses a single deterministic PCA basis per condition,
fit to the saved features and class means from seed 1001 across every checkpoint. This keeps the
display basis fixed across epochs. The interface always exposes the full-dimensional NC metrics,
the angle matrix, projection status, model, condition, epoch, and empirical/modelled boundary.

## Claim boundary

The study can establish reproducible behavior of these specified algorithms on this versioned
writer-disjoint digit source. It cannot establish causality, universal necessity or sufficiency of
neural collapse, human-level representation, robustness to natural distribution shifts, or behavior
of convolutional networks, transformers, language models, or deployed AI systems.
