# Primary-source and data audit

## Empirical source

- Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine
  Learning Repository. <https://doi.org/10.24432/C50P49>
- Official landing page: <https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2B>
- Licence: CC BY 4.0.
- Provider structure: 64 integer pixel-block counts and one digit label; training writers and test
  writers are disjoint according to the provider description.

Checksums, byte counts, row counts, class counts, and retrieval time are added to machine-readable
provenance only after the official archive is retrieved. No mirror is an authoritative fallback.

## Neural-collapse sources

1. Papyan, V., Han, X. Y. & Donoho, D. L. (2020). *Prevalence of neural collapse during the
   terminal phase of deep learning training*. PNAS 117, 24652–24663.
   <https://doi.org/10.1073/pnas.2015509117>
2. Dang, H., Tran Huu, T., Nguyen, T. M. & Ho, N. (2024). *Neural Collapse for Cross-entropy
   Class-Imbalanced Learning with Unconstrained ReLU Features Model*. ICML 2024.
   <https://proceedings.mlr.press/v235/dang24a.html>
3. Wu, D. & Mondelli, M. (2025). *Neural Collapse Beyond the Unconstrained Features Model:
   Landscape, Dynamics, and Generalization in the Mean-Field Regime*. ICML 2025.
   <https://proceedings.mlr.press/v267/wu25u.html>
4. Han, T. et al. (2025). *Flatness is Necessary, Neural Collapse is Not: Rethinking
   Generalization via Grokking*. NeurIPS 2025.
   <https://openreview.net/forum?id=lbtOctHDQ3>

The project treats the original NC1–NC4 observations as established definitions, the imbalance
paper as a condition-specific theoretical and empirical boundary, and the later generalization work
as motivation for a direct test rather than proof of the outcome expected here.
