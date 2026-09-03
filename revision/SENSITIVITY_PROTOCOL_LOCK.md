# Compact Sensitivity Protocol v1.0

Frozen on 2026-09-03 before evaluating any new sensitivity target result.

## Purpose and scope

This is a descriptive, single-seed sensitivity analysis requested during
revision. It does not replace the five-seed primary comparisons and is not used
to select or change the formal Feature+KL configuration.

- Dataset: PACS, standard four-fold leave-one-domain-out evaluation.
- Seed: 42, controlling both the class-stratified source split and optimization.
- Backbone and training: ImageNet-pretrained ResNet-50, AdamW, learning rate
  3e-4, weight decay 1e-4, cosine schedule, batch size 64, 30 epochs, 224 input.
- Checkpoint: maximum source-validation accuracy, earliest exact tie.
- Target access: exactly once after the checkpoint is fixed.
- Fixed Feature+KL defaults: lambda_f=0.10, lambda_k=0.05, T=2,
  RandAugment N=2/M=9, and a five-epoch linear consistency ramp.

## One-factor-at-a-time grid

- lambda_f: 0.05, 0.10, 0.20
- lambda_k: 0.025, 0.05, 0.10
- temperature T: 1, 2, 4
- RandAugment magnitude M: 5, 9, 13, with N=2 fixed

Only one factor changes at a time. The common default point is reused from the
audited PACS Feature+KL seed-42 formal results, leaving 32 new runs: eight
non-default settings times four held-out domains.

Target results are reported descriptively and must not be used to tune the
default configuration. The paper must label this as a single-seed analysis and
must not attach statistical significance or stability claims to these curves.
