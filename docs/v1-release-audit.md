# v1.0.0 release audit

## Scope

This audit covers the stable research product at the release commit. It does not upgrade the bounded
experiment into a universal claim about neural networks.

## Sources checked

- Official UCI Optical Digits repository and DOI record, including CC BY 4.0 terms and the documented
  writer-disjoint files.
- Papyan, Han, and Donoho (2020) for NC1–NC4 definitions.
- Dang et al. (2024) for the class-imbalance boundary.
- Wu and Mondelli (2025) for regime-specific theory connecting collapse and generalization.
- Han et al. (2025) for counterevidence to treating collapse as necessary for generalization.

Exact citations and links are in `docs/source-audit.md`.

## Reproduction executed

1. Retrieved the official UCI archive and verified the archive plus both data-member SHA-256 hashes.
2. Confirmed 3,823 official training rows, 1,797 official test rows, and every class count.
3. Regenerated all three frozen training conditions.
4. Executed 600 epochs for every condition × seed cell (30 MLP runs) and all three logistic references.
5. Regenerated `reports/results-v1.0.json`, `reports/trajectory-v1.0.json`, and the site data.
6. Verified strict finite values and strict JSON serialization; no NaN or Infinity is admitted.
7. Compared the narrative, website, and release decisions with the committed result register.

## Result decisions

- Clean NC1 after zero error: 7/10, gate not passed.
- Clean NC2 after zero error: 7/10, gate not passed.
- Clean NC4 after zero error: 10/10, gate passed.
- Long-tail NC2 worse than paired clean NC2: 10/10, gate passed.

## Deviations and numerical note

The scientific protocol was not changed after inspection. During pilot engineering, NumPy 2.2 on the
local macOS Accelerate backend emitted false floating-status warnings for demonstrably finite matrix
products. The formal runner suppresses those backend status messages only inside numerical calls and
explicitly rejects any non-finite coefficient, intercept, hidden feature, or probability state. This
engineering correction preceded the formal 30-run analysis and does not alter data, seeds, models,
epochs, checkpoints, metrics, or gates.

## Remaining limits

One dataset, one writer split, one compact MLP, one optimizer, and ten algorithmic seeds were studied.
The 2D animation is a fixed PCA projection; confirmatory NC metrics use all nine hidden dimensions.
Seeds are not treated as population samples. Label corruption is synthetic and symmetric. Full limits
are stated in `reports/research-report-v1.0.md`.

## Release identity

The exact Git commit and annotated tag are added to the public GitHub release after the release PR is
merged. The committed artifacts are immutable for v1.0.0; successors receive new versioned filenames.
