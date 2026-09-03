# ShiftGuard controlled domain-generalization study

This repository is the reproducibility package for a controlled study of how
much benefit remains after weak-to-strong consistency is separated from its
direct strong-augmentation baseline. It contains code, immutable manifests,
per-run JSON/CSV records, audit metadata, summary scripts, and figure code.
It intentionally contains no manuscript, dataset images, or model checkpoints.

## Main evidence

All formal comparisons use leave-one-domain-out evaluation. The training seed
is the replicate; domains are averaged within each seed.

| Dataset/backbone | Strong Aug. | Feature+KL | Paired delta (95% CI) |
|---|---:|---:|---:|
| PACS / ResNet-50 | 86.81 +/- 0.63 | 87.19 +/- 0.83 | +0.38 [-0.50, +1.27] |
| VLCS / ResNet-50 | 76.51 +/- 1.31 | 76.34 +/- 0.82 | -0.18 [-1.64, +1.29] |
| OfficeHome / ResNet-50 | 66.42 +/- 0.54 | 66.63 +/- 0.21 | +0.22 [-0.64, +1.07] |
| PACS / ViT-S/16 | 85.78 +/- 1.12 | 86.22 +/- 0.50 | +0.44 [-0.73, +1.60] |

Every interval contains zero. These results support a small, heterogeneous
residual effect, not universal or statistically reliable superiority.

## Data

Place datasets under these paths:

```text
data/PACS/{photo,art_painting,cartoon,sketch}/<class>/*
data/VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/<class>/*
data/OfficeHome/{Art,Clipart,Product,Real World}/<class>/*
```

Sources used by the project:

- PACS Hugging Face export: https://huggingface.co/datasets/flwrlabs/pacs
- VLCS DomainBed mirror: https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8
- OfficeHome official site: http://hemanthdv.org/OfficeHome-Dataset/
- OfficeHome DomainBed mirror: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC

PACS can be exported directly:

```bash
pip install -r requirements.txt
python download_pacs_hf.py --output data/PACS
```

Validate the uploaded VLCS and OfficeHome copies against the frozen exclusion
policy and create inventories:

```bash
python validate_revision_dataset.py --dataset vlcs --root data/VLCS \
  --inventory revision/vlcs_inventory.json --verify-images \
  --exclusions revision/cross_dataset_exclusions.json
python validate_revision_dataset.py --dataset officehome --root data/OfficeHome \
  --inventory revision/officehome_inventory.json --verify-images \
  --exclusions revision/cross_dataset_exclusions.json
```

The expected raw counts are PACS 9,991, VLCS 10,729, and OfficeHome 15,588.
The committed inventory files contain the exact fingerprints used by the
formal manifests.

## Environment

The recorded runs used Python with PyTorch 2.11.0, torchvision 0.26.0,
timm 1.0.29, CUDA 12.8, and ImageNet-pretrained models. Install the dependencies
in a fresh environment. Download and verify the exact ViT-S/16 weights with:

```bash
bash download_vit_weights.sh
```

The expected SHA-256 is
`545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3`.

## Protocol

Formal seeds are `42, 123, 3407, 2026, 2027`. In every formal run:

- the held-out target is absent from optimization, early stopping, and
  checkpoint selection;
- Strong Augmentation and Feature+KL share the split, views, transforms,
  backbone, optimizer, schedule, batch size, epochs, and checkpoint rule;
- target accuracy is evaluated once after the source-validation checkpoint is
  fixed;
- the formal Feature+KL configuration is `lambda_f=0.10`, `lambda_k=0.05`,
  `T=2`, a five-epoch ramp, and RandAugment `N=2, M=9`.

The original PACS configuration search pooled source-validation rankings over
outer folds and was not fully nested. No strict nested experiment is claimed.
The transferred Feature+KL configuration was frozen before the new VLCS,
OfficeHome, added-seed, and ViT target results were examined.

## Reproduce summaries

The committed per-run results are under `runs/`; manifests and audit summaries
are under `revision/`. Recompute all paper-level values and the 2x2 figure:

```bash
python skills/shiftguard-revision/scripts/summarize_revision.py
python skills/shiftguard-revision/scripts/summarize_cross.py
python skills/shiftguard-revision/scripts/summarize_vit.py
python summarize_compact_sensitivity.py
python summarize_revision_baselines.py
python make_revision_evidence_figure.py
```

The last command writes `figures/fig2_revision_evidence.{pdf,png}`.

## Run examples

ResNet-50 PACS direct pair for a held-out Sketch target:

```bash
python shiftguard_corrected.py --data-root data/PACS --target Sketch \
  --method aug --run-name strong_aug --model resnet50 --seed 42 \
  --epochs 30 --batch-size 64 --lr 0.0003 --weight-decay 0.0001 \
  --output runs/reproduction/strong_aug --save-checkpoint

python shiftguard_corrected.py --data-root data/PACS --target Sketch \
  --method feat_kl --run-name feature_plus_kl --model resnet50 --seed 42 \
  --epochs 30 --batch-size 64 --lr 0.0003 --weight-decay 0.0001 \
  --lambda-feat 0.10 --lambda-kl 0.05 --temperature 2 \
  --warmup-epochs 5 --output runs/reproduction/feature_plus_kl \
  --save-checkpoint
```

The frozen matrix launchers and their manifest generators are in
`skills/shiftguard-revision/scripts/`. Run their `--dry-run` modes first.
MixStyle and SWAD launchers are `run_mixstyle_matrix.py` and
`run_swad_matrix.py`; sensitivity is `run_compact_sensitivity.py`.

## Baseline disclosure

MixStyle uses the official operation with `p=0.5`, `alpha=0.1`, inserted after
ResNet layers 1 and 2. SWAD is a source-only epoch-level adaptation of the
official LossValley rule (`n_converge=3`, `n_tolerance=6`,
`tolerance_ratio=0.3`). It is not an unmodified official DomainBed launcher.
The exact upstream source files, licenses, and hashes are included under
`third_party/` and `revision/baseline_source_hashes.sha256`.

## Sensitivity disclosure

Sensitivity is a predeclared, descriptive single-seed study using seed 42 and
all four PACS outer targets. It varies one factor at a time and was not used to
retune the formal configuration. See `revision/SENSITIVITY_PROTOCOL_LOCK.md`.
