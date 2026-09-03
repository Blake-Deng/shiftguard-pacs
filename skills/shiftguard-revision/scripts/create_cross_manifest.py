#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from revision_common import FORMAL_SEEDS, canonical_hash, project_root

PROTOCOL = "shiftguard-cross-dataset-v1.1"
DATASETS = {
    "vlcs": {
        "targets": ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
        "root": "data/VLCS",
        "inventory": "revision/vlcs_inventory.json",
        "gpu": 1,
    },
    "officehome": {
        "targets": ("Art", "Clipart", "Product", "Real_World"),
        "root": "data/OfficeHome",
        "inventory": "revision/officehome_inventory.json",
        "gpu": 2,
    },
}
METHODS = {
    "erm": {"trainer_method": "erm", "lambda_f": 0.0, "lambda_k": 0.0, "views": "weak_only"},
    "strong_aug": {"trainer_method": "strong_aug", "lambda_f": 0.0, "lambda_k": 0.0, "views": "matched_weak_strong"},
    "feature_plus_kl": {"trainer_method": "feature_plus_kl", "lambda_f": 0.10, "lambda_k": 0.05, "views": "matched_weak_strong"},
}
FIELDS = [
    "experiment_id", "experiment_group", "protocol_version", "dataset", "data_root",
    "outer_target", "method", "trainer_method", "backbone", "seed", "optimization_seed",
    "source_split_seed", "lambda_f", "lambda_k", "temperature", "augmentation_M",
    "selection_protocol", "config_hash", "dataset_fingerprint", "status", "result_path",
    "checkpoint_path", "gpu", "exit_code", "rerun_reason",
]


def scientific_config(
    dataset: str, target: str, method: str, seed: int, fingerprint: str,
    exclusion_policy: str, excluded_images: int,
) -> dict:
    method_spec = METHODS[method]
    return {
        "protocol_version": PROTOCOL,
        "dataset": dataset,
        "outer_target": target,
        "method": method,
        "trainer_method": method_spec["trainer_method"],
        "views": method_spec["views"],
        "backbone": "torchvision:resnet50:IMAGENET1K_V2",
        "preprocessing": "legacy_imagenet",
        "seed": seed,
        "optimization_seed": seed,
        "source_split_seed": seed,
        "source_val_fraction": 0.15,
        "epochs": 30,
        "batch_size": 64,
        "image_size": 224,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "lambda_f": method_spec["lambda_f"],
        "lambda_k": method_spec["lambda_k"],
        "temperature": 2.0,
        "consistency_ramp_epochs": 5,
        "augmentation_N": 2,
        "augmentation_M": 9,
        "checkpoint_rule": "highest_source_validation_accuracy_earliest_exact_tie",
        "target_evaluations": 1,
        "dataset_fingerprint": fingerprint,
        "exclusion_policy": exclusion_policy,
        "excluded_images": excluded_images,
    }


def main() -> None:
    root = project_root()
    rows = []
    for dataset, dataset_spec in DATASETS.items():
        inventory = json.loads((root / dataset_spec["inventory"]).read_text(encoding="utf-8"))
        fingerprint = inventory["dataset_fingerprint"]
        exclusion_policy = inventory["exclusion_policy"]["policy_version"]
        excluded_images = inventory["excluded_image_count"]
        for method in METHODS:
            for target in dataset_spec["targets"]:
                for seed in FORMAL_SEEDS:
                    config = scientific_config(
                        dataset, target, method, seed, fingerprint,
                        exclusion_policy, excluded_images,
                    )
                    experiment_id = f"cross-{dataset}-{method}-{target}-seed{seed}"
                    output_dir = root / "runs" / "revision" / "cross_dataset" / dataset / method / target / str(seed)
                    stem = f"{dataset}_{target}_resnet50_{method}_seed{seed}"
                    rows.append({
                        "experiment_id": experiment_id,
                        "experiment_group": "P0_cross_dataset_confirmation",
                        "protocol_version": PROTOCOL,
                        "dataset": dataset,
                        "data_root": dataset_spec["root"],
                        "outer_target": target,
                        "method": method,
                        "trainer_method": METHODS[method]["trainer_method"],
                        "backbone": config["backbone"],
                        "seed": seed,
                        "optimization_seed": seed,
                        "source_split_seed": seed,
                        "lambda_f": config["lambda_f"],
                        "lambda_k": config["lambda_k"],
                        "temperature": config["temperature"],
                        "augmentation_M": config["augmentation_M"],
                        "selection_protocol": config["checkpoint_rule"],
                        "config_hash": canonical_hash(config),
                        "dataset_fingerprint": fingerprint,
                        "status": "pending",
                        "result_path": str((output_dir / f"{stem}.json").relative_to(root)),
                        "checkpoint_path": str((output_dir / f"{stem}.pt").relative_to(root)),
                        "gpu": str(dataset_spec["gpu"]),
                        "exit_code": "",
                        "rerun_reason": "",
                    })
    manifest = root / "revision" / "cross_dataset_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if {row["experiment_id"] for row in existing} != {row["experiment_id"] for row in rows}:
            raise RuntimeError("existing cross-dataset manifest IDs differ; refusing to overwrite")
        if any(row["status"] != "pending" for row in existing):
            raise RuntimeError("existing manifest has launched jobs; refusing to overwrite")
        print(f"Refreshing {len(existing)} pending jobs for protocol {PROTOCOL}: {manifest}")
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    print(json.dumps({
        "manifest": str(manifest),
        "jobs": len(rows),
        "by_dataset": {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in DATASETS},
    }, indent=2))


if __name__ == "__main__":
    main()
