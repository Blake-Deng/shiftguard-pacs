# ShiftGuard PACS Domain Generalization

Source code for the ShiftGuard PACS domain-generalization experiments. This repository intentionally contains no manuscript, figures, local PACS images, checkpoints, or pretrained weights.

## Dataset

Use the complete PACS dataset from Hugging Face:

- Dataset page: https://huggingface.co/datasets/flwrlabs/pacs
- Dataset identifier: `flwrlabs/pacs`

The expected exported layout is:

```text
data/PACS/{Photo,Art_Painting,Cartoon,Sketch}/{dog,elephant,giraffe,guitar,horse,person,house}/*.jpg
```

Download and export it with:

```bash
python -m pip install -U datasets
python download_pacs_hf.py --output data/PACS
python check_pacs.py --data-root data/PACS
```

The Hugging Face dataset is the data source. Do not commit the downloaded images to GitHub; add `data/` to `.gitignore`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For the exact ViT-S/16 experiment, place the local ImageNet checkpoint at `weights/vit_small_patch16_224.npz` or use the checkpoint-loading configuration documented in `shiftguard_v2.py`. ResNet-50 weights are obtained by torchvision when `--pretrained` is enabled.

## Quick smoke test

```bash
./run_smoke.sh /path/to/data/PACS
```

## Main experiments

The canonical implementation is `shiftguard_v2.py`. It supports ERM, strong augmentation, Mixup, feature consistency, feature+KL consistency, and ShiftGuard.

```bash
python shiftguard_v2.py --data-root data/PACS --target Sketch \\
  --method shiftguard --model resnet50 --seed 42 --epochs 30 \\
  --lambda-feat 0.5 --lambda-kl 0.25 --output runs
```

Run the three-seed, four-target matrix:

```bash
./run_main_matrix.sh /path/to/data/PACS
python summarize_results.py runs/results.csv
```

Additional scripts:

- `coral_exp.py`: CORAL baseline.
- `robustness_eval.py`: Sketch-target corruption evaluation.
- `make_figures.py`: optional result visualization.
- `shiftguard.py`: minimal ResNet-50 implementation used for the first baseline matrix.
- `shiftguard_exp.py`: extended experiment entry point, including Mixup and exact timm ViT-S/16 support.

## Reproducibility notes

Each target domain is held out completely. Source images are split into training and validation subsets; the best source-validation checkpoint is restored before target evaluation. The reported matrix uses targets `Photo`, `Art_Painting`, `Cartoon`, and `Sketch`, and seeds `42`, `123`, and `3407`.

## License and data notice

Add the license required by your institution before publishing this repository. PACS images remain subject to their original dataset terms; link to the Hugging Face dataset instead of redistributing them.
