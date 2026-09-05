# ShiftGuard-CGC reproducibility package

This repository contains code, immutable protocols, manifests, per-run results,
and summary utilities for source-only component-wise conflict-gated consistency
(CGC) in visual domain generalization. It intentionally contains no manuscript,
LaTeX source, paper PDF, dataset images, or model checkpoints.

## Included artifacts

- `cgc_experiment.py`: matched Strong Augmentation, Feature+KL, CGC, Mean
  Teacher, and gate ablations.
- `run_cgc_complete_queue.py`: restartable multi-GPU queue and result audit.
- `summarize_cgc_results.py`: five-seed statistics, NTR, nested selection,
  mechanism diagnostics, and efficiency summaries.
- `plot_cgc_evidence.py`: quantitative 2x2 result visualization.
- `summarize_cgc_results_offline.py`: checkpoint-free summary entry point that
  reuses the committed inference-latency snapshot while recomputing all other
  values from per-run JSON.
- `revision/CGC_PROTOCOL_LOCK.md`: protocol frozen before formal target results.
- `revision/cgc_v2_manifest.csv`: all 232 new jobs and completion records.
- `runs/cgc_v2/`: 243 new per-run JSON files, without checkpoints.
- `results/cgc_diagnostics/`: 143 epoch-level diagnostic JSON files.
- Prior control JSON under `runs/revision/` and `runs/corrected_ablation/`,
  required to recompute Strong Augmentation and Feature+KL comparisons.

## Recorded results

Formal comparisons use seeds `42`, `123`, `3407`, `2026`, and `2027`. The
training seed is the replicate; domains are averaged within each seed.

| Method | PACS | VLCS | OfficeHome |
|---|---:|---:|---:|
| Strong Augmentation | 86.81 +/- 0.63 | 76.51 +/- 1.31 | 66.42 +/- 0.54 |
| Feature+KL | 87.19 +/- 0.83 | 76.34 +/- 0.82 | 66.63 +/- 0.21 |
| CGC | 86.96 +/- 0.63 | 76.87 +/- 0.38 | 66.37 +/- 0.38 |
| MixStyle | 83.77 +/- 0.96 | 75.39 +/- 1.33 | 64.34 +/- 0.54 |
| SWAD (epoch) | 84.25 +/- 1.34 | 78.40 +/- 0.55 | 65.98 +/- 0.33 |

CGC minus Strong Augmentation paired means and 95% t intervals are PACS
`+0.15 [-0.46, +0.76]`, VLCS `+0.35 [-1.60, +2.31]`, OfficeHome
`-0.04 [-1.02, +0.94]`, and PACS ViT-S/16 `-0.51 [-1.92, +0.91]` percentage
points. All intervals contain zero. The aggregate NTR is 48.3% for Feature+KL
and 45.0% for CGC. Strict nested PACS selection obtains `87.24 +/- 0.60`.

## Data

Place datasets under:

```text
data/PACS/{photo,art_painting,cartoon,sketch}/<class>/*
data/VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/<class>/*
data/OfficeHome/{Art,Clipart,Product,Real World}/<class>/*
```

Dataset sources:

- PACS: https://huggingface.co/datasets/flwrlabs/pacs
- VLCS DomainBed mirror: https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8
- OfficeHome official site: http://hemanthdv.org/OfficeHome-Dataset/
- OfficeHome DomainBed mirror: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC

Expected raw counts are PACS 9,991, VLCS 10,729, and OfficeHome 15,588. Exact
inventories and the image-exclusion policy are under `revision/`. PACS can be
exported with `python download_pacs_hf.py --output data/PACS`.

## Environment

Recorded versions are PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.29, and
CUDA 12.8. Install dependencies and the verified ViT weights with:

```bash
pip install -r requirements.txt
bash download_vit_weights.sh
```

## Target-isolation protocol

For every outer target, target images are absent from optimization, early
stopping, checkpoint selection, and configuration selection. Formal JSON files
record `target_evaluations = 1` only after checkpoint restoration. Strict nested
screening records `target_evaluations = 0`, `target_accuracy = null`, and
`n_test = 0`. CGC uses a fixed zero threshold and detached feature/KL masks.
Matched methods share splits, views, backbone, optimizer, schedule, batch size,
epochs, and checkpoint rule.

## Verify the release

After installing dependencies:

```bash
python cgc_queue_status.py
python summarize_cgc_results_offline.py
python plot_cgc_evidence.py
```

Expected queue status is 232 complete and zero failed. Summaries are written to
`revision/cgc_v2_summaries/`, and the plot is written to `figures/`.

## Run one CGC experiment

```bash
python cgc_experiment.py \
  --dataset pacs --data-root data/PACS --target Sketch \
  --method cgc --run-name reproduce_pacs_sketch_cgc_seed42 \
  --model resnet50 --seed 42 --epochs 30 --batch-size 64 \
  --lambda-feat 0.10 --lambda-kl 0.05 --temperature 2 \
  --warmup-epochs 5 --augmentation-m 9 \
  --exclusions revision/cross_dataset_exclusions.json \
  --output runs/reproduction/pacs_sketch_cgc_seed42 \
  --diagnostics-output results/reproduction/pacs_sketch_cgc_seed42 \
  --save-checkpoint
```

The full queue launcher is restartable. Check it before launching new jobs:

```bash
python run_cgc_complete_queue.py --gpus 0,1,2 --workers-per-gpu 4 --dry-run
```

## Baseline disclosure

MixStyle uses `p=0.5`, `alpha=0.1` after ResNet layers 1 and 2. SWAD is a
source-only epoch-level LossValley adaptation under the matched schedule, not
an unmodified official DomainBed launcher. Upstream source, licenses, and
hashes are retained under `third_party/` and `revision/`.

This repository preserves mixed and negative outcomes. It must not be used to
claim statistically significant or universal superiority when paired intervals
include zero.
