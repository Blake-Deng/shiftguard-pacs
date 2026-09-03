#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import timm
from timm.data import create_transform, resolve_model_data_config

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import shiftguard_corrected as trainer
from revision_common import file_sha256, pacs_inventory, project_root


def main() -> None:
    root = project_root()
    fingerprint, inventory = pacs_inventory(root / "data" / "PACS")
    weights = root / "weights" / "vit_small_patch16_224.npz"
    if not weights.is_file():
        raise RuntimeError("missing ViT-S/16 pretrained weights")
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    cfg = resolve_model_data_config(model)
    expected = {
        "input_size": (3, 224, 224),
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "crop_pct": 0.9,
    }
    for key, value in expected.items():
        if cfg[key] != value:
            raise RuntimeError(f"timm metadata mismatch for {key}: {cfg[key]!r}")
    evaluation = trainer.make_transforms(224, "timm_vit_standard")[2]
    reference = create_transform(**cfg, is_training=False)
    if "Resize(size=248, interpolation=bicubic" not in repr(evaluation):
        raise RuntimeError(f"unexpected trainer evaluation transform: {evaluation}")
    if "Normalize(mean=(0.5, 0.5, 0.5)" not in repr(evaluation):
        raise RuntimeError(f"unexpected trainer normalization: {evaluation}")
    free_gib = shutil.disk_usage(root).free / (1024 ** 3)
    if free_gib < 20:
        raise RuntimeError(f"only {free_gib:.1f} GiB free")
    gpus = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    print(json.dumps({
        "status": "PASS",
        "dataset_images": len(inventory),
        "dataset_fingerprint": fingerprint,
        "weights_sha256": file_sha256(weights),
        "timm_data_config": cfg,
        "reference_eval_transform": repr(reference),
        "trainer_eval_transform": repr(evaluation),
        "free_disk_gib": round(free_gib, 1),
        "gpus": gpus,
    }, indent=2, default=list))


if __name__ == "__main__":
    main()
