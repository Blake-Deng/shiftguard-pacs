#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import shiftguard_multidataset as trainer
import validate_revision_dataset as dataset_validator
from create_cross_manifest import DATASETS


def main() -> None:
    report = {"status": "PASS", "datasets": {}, "source_only_checks": []}
    exclusions_path = ROOT / "revision" / "cross_dataset_exclusions.json"
    for dataset, spec in DATASETS.items():
        inventory = json.loads((ROOT / spec["inventory"]).read_text(encoding="utf-8"))
        data_root = ROOT / spec["root"]
        if not data_root.is_dir():
            raise RuntimeError(f"missing dataset root: {data_root}")
        current = dataset_validator.validate(dataset, data_root, True, exclusions_path)
        if current["dataset_fingerprint"] != inventory["dataset_fingerprint"]:
            raise RuntimeError(f"{dataset} inventory fingerprint is stale")
        excluded, _ = trainer.exclusion_paths(exclusions_path, dataset)
        report["datasets"][dataset] = {
            "root": str(data_root),
            "images": inventory["image_count"],
            "classes": inventory["class_count"],
            "fingerprint": inventory["dataset_fingerprint"],
            "excluded_images": inventory["excluded_image_count"],
        }
        for target in spec["targets"]:
            train, val, _, sources = trainer.collect_source_samples(data_root, dataset, target, 42, 0.15, excluded)
            target_dir = trainer.resolve_domains(data_root, dataset)[target].resolve()
            leaked = [item[0] for item in train + val if Path(item[0]).resolve().is_relative_to(target_dir)]
            if leaked:
                raise RuntimeError(f"target samples leaked into source split for {dataset}/{target}")
            report["source_only_checks"].append({
                "dataset": dataset,
                "target": target,
                "sources": sources,
                "n_train": len(train),
                "n_val": len(val),
                "target_samples_in_split": 0,
            })
    if not (Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth").is_file():
        raise RuntimeError("missing cached ResNet-50 pretrained weights")
    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gib < 30:
        raise RuntimeError(f"only {free_gib:.1f} GiB free")
    report["free_disk_gib"] = round(free_gib, 1)
    report["gpus"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
