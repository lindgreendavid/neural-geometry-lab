# Primary-source and data audit

## Empirical source

- Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine
  Learning Repository. <https://doi.org/10.24432/C50P49>
- Official landing page: <https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2B>
- Licence: CC BY 4.0.
- Provider structure: 64 integer pixel-block counts and one digit label; training writers and test
  writers are disjoint according to the provider description.

The official archive was retrieved on 2026-08-14 at 19:54:04 UTC. The immutable audit is stored in
`data/provenance-v1.0.json`. Its verified identities are:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| Official ZIP | 591,292 | `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0` |
| `optdigits.tra` | 563,639 | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| `optdigits.tes` | 264,712 | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |

The loader confirms 3,823 training rows and 1,797 test rows. Training class counts for digits
0–9 are 376, 389, 380, 389, 387, 376, 377, 387, 380, and 382; test counts are 178, 182,
177, 183, 181, 182, 181, 179, 174, and 180. No mirror was used.

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
