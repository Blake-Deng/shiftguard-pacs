#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from revision_common import DOMAINS, pacs_inventory, project_root


def validate_old_result(path: Path, method: str) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("target_evaluations") != 1 or result.get("target_accuracy") is None:
        raise RuntimeError(f"invalid target audit in {path}")
    config = result.get("config", {})
    expected_f = 0.0 if method == "strong_aug" else 0.10
    expected_k = 0.0 if method == "strong_aug" else 0.05
    if abs(float(config.get("lambda_feat", -1)) - expected_f) > 1e-12:
        raise RuntimeError(f"lambda_f mismatch in {path}")
    if abs(float(config.get("lambda_kl", -1)) - expected_k) > 1e-12:
        raise RuntimeError(f"lambda_k mismatch in {path}")
    if float(config.get("temperature", -1)) != 2.0 or int(config.get("warmup_epochs", -1)) != 5:
        raise RuntimeError(f"temperature/ramp mismatch in {path}")


def main() -> None:
    root = project_root()
    fingerprint, inventory = pacs_inventory(root / "data" / "PACS")
    checked = 0
    for method in ("strong_aug", "feature_plus_kl"):
        for domain in DOMAINS:
            for seed in (42, 123, 3407):
                path = root / "runs" / "corrected_ablation" / "tasks" / method / domain / str(seed) / f"{domain}_resnet50_{method}_seed{seed}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing historical result: {path}")
                validate_old_result(path, method)
                checked += 1
    if not (root / "resnet50-11ad3fa6.pth").is_file():
        raise RuntimeError("missing local ResNet-50 pretrained weights")
    free_gib = shutil.disk_usage(root).free / (1024 ** 3)
    if free_gib < 20:
        raise RuntimeError(f"only {free_gib:.1f} GiB free")
    gpu_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    print(json.dumps({
        "status": "PASS",
        "pacs_images": len(inventory),
        "pacs_fingerprint": fingerprint,
        "historical_results_validated": checked,
        "free_disk_gib": round(free_gib, 1),
        "gpus": gpu_text,
    }, indent=2))


if __name__ == "__main__":
    main()
