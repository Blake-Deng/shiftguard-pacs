# ShiftGuard PACS corrected experiments

This repository contains the corrected, target-blind PACS experiment code.
It intentionally excludes the manuscript, author information, PACS images,
model checkpoints, and pretrained weights.

## Dataset

- Hugging Face: https://huggingface.co/datasets/flwrlabs/pacs
- Identifier: flwrlabs/pacs
- Size: 9,991 images, four domains, seven classes

Expected layout:

    data/PACS/{photo,art_painting,cartoon,sketch}/
      {dog,elephant,giraffe,guitar,horse,house,person}/*.jpg

Download and validate:

    python -m pip install -r requirements.txt
    python download_pacs_hf.py --output data/PACS
    python check_pacs.py data/PACS

Do not commit data/PACS. PACS remains subject to its original terms.

## Environment

Reported environment: PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
timm 1.0.29, CUDA 12.8, and NVIDIA RTX 5090.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -r requirements.txt

ResNet-50 weights are downloaded by torchvision. For exact pretrained
ViT-S/16:

    bash download_vit_weights.sh

The script writes weights/vit_small_patch16_224.npz and verifies SHA-256:
545815b4e770d2fa6ca4b3ccba7c16b035e474354e52d17ca197ea4efecbf4d3.

## Corrected implementation

shiftguard_corrected.py supports:

- aug: matched weak/strong classification baseline
- kl: one-way weak-to-strong KL
- feat: detached feature consistency
- feat_kl: detached feature plus one-way KL
- adaptive: Feature+KL with detached reliability weighting

Screening does not construct the target dataset. Formal evaluation restores
the best source-validation checkpoint, then constructs and evaluates the
target exactly once.

Single-run example:

    python shiftguard_corrected.py \
      --data-root data/PACS \
      --target Sketch \
      --method feat_kl \
      --run-name feature_plus_kl \
      --model resnet50 \
      --seed 42 \
      --epochs 30 \
      --batch-size 64 \
      --lambda-feat 0.10 \
      --lambda-kl 0.05 \
      --temperature 2.0 \
      --gate-tau 0.5 \
      --warmup-epochs 5 \
      --save-checkpoint \
      --output runs/example

Smoke test:

    python shiftguard_corrected.py \
      --data-root data/PACS --target Sketch --method aug \
      --run-name smoke --seed 42 --epochs 1 --batch-size 16 \
      --workers 2 --no-pretrained --output runs/smoke

## Experiment order

One GPU is the default. To select GPUs:

    export SHIFTGUARD_GPUS=0

or:

    export SHIFTGUARD_GPUS=0,1,2

1. Source-only screening, formal ResNet, and exact ViT-S/16:

    python run_corrected_pipeline.py

2. Four variants x four targets x three seeds:

    python run_corrected_ablation.py

3. After ablation checkpoints exist, Sketch corruption robustness:

    bash run_robustness_corrected.sh

## Baseline context

baselines contains the ERM, Mixup, and source-domain CORAL implementations
used for contextual comparisons.

Mixup:

    python baselines/shiftguard_exp.py \
      --data-root data/PACS --target Sketch --method mixup \
      --model resnet50 --seed 42 --epochs 30 --output runs/mixup

Source-only CORAL:

    python baselines/coral_exp.py \
      --data-root data/PACS --target Sketch --method coral \
      --model resnet50 --seed 42 --epochs 30 \
      --lambda-feat 0.5 --output runs/coral

CORAL aligns covariance among source domains only.

## Reference results

reference_results includes every lightweight per-run JSON record:

- corrected_ablation: 48 runs and summary
- corrected_formal: 12 runs and summary
- corrected_vit: 24 runs and summary
- corrected_screening: source-validation selection records
- corrected_robustness: 150 evaluations

Macro standard deviation is calculated after averaging four domains within
each seed, then taking sample standard deviation across three seed macros.

## Interpretation

Strong Augmentation: 87.20 +/- 0.16 percent.
Feature+KL: 87.46 +/- 1.01 percent, a descriptive +0.26 point difference.
Adaptive: 86.93 +/- 1.28 percent and not uniformly better.

Do not claim significance from four domains or treat corruption severity
levels as independent training runs.
