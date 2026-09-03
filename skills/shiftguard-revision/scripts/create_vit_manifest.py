#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from revision_common import DOMAINS, FORMAL_SEEDS, canonical_hash, file_sha256, pacs_inventory, project_root

PROTOCOL = "shiftguard-vit-direct-v1.0"
FIELDS = [
    "experiment_id", "experiment_group", "protocol_version", "dataset", "outer_target",
    "method", "trainer_method", "backbone", "trainer_model", "preprocessing", "seed",
    "optimization_seed", "source_split_seed", "lambda_f", "lambda_k", "temperature",
    "augmentation_M", "selection_protocol", "pretrained_weights_sha256", "config_hash",
    "dataset_fingerprint", "status", "result_path", "checkpoint_path", "gpu", "exit_code",
    "rerun_reason",
]


def config_for(method: str, target: str, seed: int, dataset_fingerprint: str, weights_sha256: str) -> dict:
    return {
        "protocol_version": PROTOCOL,
        "dataset": "PACS",
        "outer_target": target,
        "train_domains": [domain for domain in DOMAINS if domain != target],
        "method": method,
        "trainer_method": "aug" if method == "strong_aug" else "feat_kl",
        "backbone": "timm:vit_small_patch16_224.augreg_in21k_ft_in1k",
        "trainer_model": "vit-small",
        "pretrained_weights_sha256": weights_sha256,
        "preprocessing": "timm_vit_standard",
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5],
        "interpolation": "bicubic",
        "evaluation_resize": 248,
        "evaluation_crop": 224,
        "seed": seed,
        "optimization_seed": seed,
        "source_split_seed": seed,
        "source_val_fraction": 0.15,
        "epochs": 30,
        "batch_size": 64,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "lambda_f": 0.0 if method == "strong_aug" else 0.10,
        "lambda_k": 0.0 if method == "strong_aug" else 0.05,
        "temperature": 2.0,
        "consistency_ramp_epochs": 5,
        "augmentation_N": 2,
        "augmentation_M": 9,
        "checkpoint_rule": "highest_source_validation_accuracy_earliest_exact_tie",
        "target_evaluations": 1,
        "dataset_fingerprint": dataset_fingerprint,
    }


def main() -> None:
    root = project_root()
    dataset_fingerprint, _ = pacs_inventory(root / "data" / "PACS")
    weights = root / "weights" / "vit_small_patch16_224.npz"
    if not weights.is_file():
        raise RuntimeError(f"missing pretrained weights: {weights}")
    weights_sha256 = file_sha256(weights)
    rows = []
    for method in ("strong_aug", "feature_plus_kl"):
        for target in DOMAINS:
            for seed in FORMAL_SEEDS:
                config = config_for(method, target, seed, dataset_fingerprint, weights_sha256)
                experiment_id = f"vit-direct-{method}-{target}-seed{seed}"
                output_dir = root / "runs" / "revision" / "vit_direct" / method / target / str(seed)
                stem = f"{target}_vit-small_{method}_seed{seed}"
                rows.append({
                    "experiment_id": experiment_id,
                    "experiment_group": "P0_exact_ViT_S16_direct_comparison",
                    "protocol_version": PROTOCOL,
                    "dataset": "PACS",
                    "outer_target": target,
                    "method": method,
                    "trainer_method": config["trainer_method"],
                    "backbone": config["backbone"],
                    "trainer_model": "vit-small",
                    "preprocessing": config["preprocessing"],
                    "seed": seed,
                    "optimization_seed": seed,
                    "source_split_seed": seed,
                    "lambda_f": config["lambda_f"],
                    "lambda_k": config["lambda_k"],
                    "temperature": config["temperature"],
                    "augmentation_M": config["augmentation_M"],
                    "selection_protocol": config["checkpoint_rule"],
                    "pretrained_weights_sha256": weights_sha256,
                    "config_hash": canonical_hash(config),
                    "dataset_fingerprint": dataset_fingerprint,
                    "status": "pending",
                    "result_path": str((output_dir / f"{stem}.json").relative_to(root)),
                    "checkpoint_path": str((output_dir / f"{stem}.pt").relative_to(root)),
                    "gpu": "",
                    "exit_code": "",
                    "rerun_reason": "",
                })
    manifest = root / "revision" / "vit_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if {row["experiment_id"] for row in existing} != {row["experiment_id"] for row in rows}:
            raise RuntimeError("existing ViT manifest IDs differ; refusing to overwrite")
        print(f"ViT manifest already exists with {len(existing)} rows: {manifest}")
        return
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    print(json.dumps({"manifest": str(manifest), "jobs": len(rows), "weights_sha256": weights_sha256}, indent=2))


if __name__ == "__main__":
    main()
