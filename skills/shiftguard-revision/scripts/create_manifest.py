#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from revision_common import DOMAINS, NEW_SEEDS, PROTOCOL_VERSION, canonical_hash, pacs_inventory, project_root, scientific_config

FIELDS = [
    "experiment_id", "experiment_group", "protocol_version", "dataset", "outer_target",
    "method", "trainer_method", "backbone", "seed", "optimization_seed", "source_split_seed",
    "lambda_f", "lambda_k", "temperature", "augmentation_M", "selection_protocol",
    "config_hash", "dataset_fingerprint", "status", "result_path", "checkpoint_path",
    "gpu", "exit_code", "rerun_reason",
]


def main() -> None:
    root = project_root()
    revision_root = root / "revision"
    revision_root.mkdir(parents=True, exist_ok=True)
    fingerprint, inventory = pacs_inventory(root / "data" / "PACS")
    inventory_path = revision_root / "pacs_inventory.json"
    inventory_payload = {"dataset": "PACS", "count": len(inventory), "fingerprint": fingerprint, "files": inventory}
    inventory_path.write_text(json.dumps(inventory_payload, indent=2), encoding="utf-8")

    rows = []
    for method in ("strong_aug", "feature_plus_kl"):
        for target in DOMAINS:
            for seed in NEW_SEEDS:
                config = scientific_config(method, target, seed, fingerprint)
                config_hash = canonical_hash(config)
                experiment_id = f"pacs5-{method}-{target}-seed{seed}"
                output_dir = root / "runs" / "revision" / "pacs5" / method / target / str(seed)
                stem = f"{target}_resnet50_{method}_seed{seed}"
                rows.append({
                    "experiment_id": experiment_id,
                    "experiment_group": "P0_PACS_five_seed_confirmation",
                    "protocol_version": PROTOCOL_VERSION,
                    "dataset": "PACS",
                    "outer_target": target,
                    "method": method,
                    "trainer_method": config["trainer_method"],
                    "backbone": "resnet50",
                    "seed": seed,
                    "optimization_seed": seed,
                    "source_split_seed": seed,
                    "lambda_f": config["lambda_f"],
                    "lambda_k": config["lambda_k"],
                    "temperature": config["temperature"],
                    "augmentation_M": config["augmentation_M"],
                    "selection_protocol": config["checkpoint_rule"],
                    "config_hash": config_hash,
                    "dataset_fingerprint": fingerprint,
                    "status": "pending",
                    "result_path": str((output_dir / f"{stem}.json").relative_to(root)),
                    "checkpoint_path": str((output_dir / f"{stem}.pt").relative_to(root)),
                    "gpu": "",
                    "exit_code": "",
                    "rerun_reason": "",
                })

    manifest = revision_root / "revision_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        existing_ids = {row["experiment_id"] for row in existing}
        new_ids = {row["experiment_id"] for row in rows}
        if existing_ids != new_ids:
            raise RuntimeError("existing manifest experiment IDs differ; refusing to overwrite")
        print(f"Manifest already exists with {len(existing)} rows: {manifest}")
        return
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    print(json.dumps({"manifest": str(manifest), "jobs": len(rows), "dataset_fingerprint": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
