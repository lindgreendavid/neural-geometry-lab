# Neural Geometry Lab

> Does a neural network's hidden geometry become simpler after perfect training accuracy—and does
> that simplicity guarantee better generalization?

Neural Geometry Lab is a frozen, reproducible comparison of multinomial logistic regression and a
small ReLU network on the official writer-disjoint UCI Optical Digits split. It measures all four
classical neural-collapse properties, then stress-tests their interpretation under class imbalance
and label noise.

Status: protocol frozen; empirical results pending.

Read the [frozen protocol](docs/protocol-v1.0.md) and [source audit](docs/source-audit.md).

## Scientific boundary

This is a controlled study of specified small models on one public digit source. The animated
feature geometry is not a claim about brains, foundation models, causal representation learning, or
deployed systems. Reported endpoints are computed in the full nine-dimensional feature space.

## Licence

Original software and documentation are MIT licensed. UCI data retain CC BY 4.0 attribution.
