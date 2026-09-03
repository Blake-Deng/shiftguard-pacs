# ShiftGuard Revision Status

Last updated: 2026-09-03

## Completed formal evidence

All runs listed below completed without recorded failures. Every formal JSON
reports `target_evaluations=1`, and the held-out target is absent from its
`source_domains` field.

### PACS five-seed confirmation

Formal seeds: `42, 123, 3407, 2026, 2027`.

| Method | Photo | Art Painting | Cartoon | Sketch | Macro |
|---|---:|---:|---:|---:|---:|
| Strong Augmentation | 96.41 +/- 0.67 | 84.45 +/- 1.95 | 82.91 +/- 1.76 | 83.47 +/- 1.97 | 86.81 +/- 0.63 |
| Feature+KL | 96.53 +/- 0.20 | 83.24 +/- 1.90 | 83.91 +/- 1.38 | 85.10 +/- 0.92 | 87.19 +/- 0.83 |
| Difference | +0.12 | -1.21 | +1.00 | +1.63 | +0.38 pp |

Paired 95% CI: `[-0.50, +1.27]` pp. It contains zero.

### Cross-dataset five-seed confirmation

| Dataset | ERM | Strong Aug. | Feature+KL | Paired delta (95% CI) |
|---|---:|---:|---:|---:|
| VLCS | 74.80 +/- 0.93 | 76.51 +/- 1.31 | 76.34 +/- 0.82 | -0.18 [-1.64, +1.29] |
| OfficeHome | 63.69 +/- 0.53 | 66.42 +/- 0.54 | 66.63 +/- 0.21 | +0.22 [-0.64, +1.07] |

Manifest: `revision/cross_dataset_manifest.csv` (120/120 complete).
Protocol: `shiftguard-cross-dataset-v1.1`.

### MixStyle and SWAD baselines

| Method | PACS | VLCS | OfficeHome |
|---|---:|---:|---:|
| MixStyle | 83.77 +/- 0.96 | 75.39 +/- 1.33 | 64.34 +/- 0.54 |
| SWAD (epoch) | 84.25 +/- 1.34 | 78.40 +/- 0.55 | 65.98 +/- 0.33 |

Both matrices contain 60/60 complete runs. SWAD is explicitly an epoch-level,
source-only adaptation of the official LossValley rule under the common
trainer, not an unmodified official DomainBed launcher.

### Exact ViT-S/16 direct comparison

Backbone: `timm vit_small_patch16_224.augreg_in21k_ft_in1k`.

| Method | Photo | Art Painting | Cartoon | Sketch | Macro |
|---|---:|---:|---:|---:|---:|
| Strong Augmentation | 96.57 +/- 0.48 | 83.50 +/- 1.32 | 83.16 +/- 1.48 | 79.89 +/- 2.39 | 85.78 +/- 1.12 |
| Feature+KL | 96.07 +/- 0.42 | 83.74 +/- 0.75 | 83.83 +/- 0.95 | 81.23 +/- 2.02 | 86.22 +/- 0.50 |

Paired delta: `+0.44` pp, 95% CI `[-0.73, +1.60]`. The manifest contains
40/40 complete runs.

### Descriptive compact sensitivity

The predeclared PACS study uses seed 42 and all four outer targets. It is
one-factor-at-a-time and was not used to retune the fixed formal configuration.

- `lambda_f` target macro range: 86.35--88.36
- `lambda_k` target macro range: 86.35--87.36
- temperature target macro range: 86.35--87.53
- RandAugment magnitude target macro range: 86.19--86.35
- source-validation macro range across all settings: 97.93--98.20

The sensitivity manifest contains 32/32 complete non-default runs; the common
default is reused from the audited PACS seed-42 result.

## Submitted evidence retained

The submitted three seeds are a subset of the five-seed PACS confirmation.
The original ERM, Mixup, source-only CORAL, One-way KL, Feature+KL, Adaptive,
and Sketch corruption records remain available. They are marked as three-seed
or exploratory where reported.

## Explicitly not completed

Strict nested source-domain configuration selection was not run. The submitted
pooled cross-fold procedure is not fully nested because a domain held out in
one fold can be a source in another fold. The revision removes claims of
strictly target-blind configuration selection and keeps this as a limitation.
Within every individual formal run, the held-out target is still absent from
optimization, early stopping, and checkpoint selection.

## Claim lock

Allowed conclusion: strong augmentation explains most of the gain over ERM;
Feature+KL has a small, heterogeneous residual effect whose paired confidence
intervals contain zero and whose direction depends on the dataset.

Do not claim statistical significance, universal superiority, state of the
art, architecture-independent benefit, or completion of strict nested model
selection.
