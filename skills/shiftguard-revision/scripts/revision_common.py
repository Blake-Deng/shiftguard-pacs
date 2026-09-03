#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
DOMAIN_DIRS = {
    "Photo": "photo",
    "Art_Painting": "art_painting",
    "Cartoon": "cartoon",
    "Sketch": "sketch",
}
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FORMAL_SEEDS = (42, 123, 3407, 2026, 2027)
NEW_SEEDS = (2026, 2027)
PROTOCOL_VERSION = "shiftguard-revision-v1.0"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pacs_inventory(data_root: Path) -> tuple[str, list[dict]]:
    rows = []
    for domain in DOMAINS:
        domain_root = data_root / DOMAIN_DIRS[domain]
        if not domain_root.is_dir():
            raise RuntimeError(f"missing PACS domain directory: {domain_root}")
        actual_classes = tuple(sorted(p.name for p in domain_root.iterdir() if p.is_dir()))
        if actual_classes != CLASSES:
            raise RuntimeError(f"class mismatch in {domain}: {actual_classes}")
        for class_name in CLASSES:
            for path in sorted((domain_root / class_name).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    rows.append({
                        "path": path.relative_to(data_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "domain": domain,
                        "class": class_name,
                    })
    if len(rows) != 9991:
        raise RuntimeError(f"expected 9991 PACS images, found {len(rows)}")
    return canonical_hash({"dataset": "PACS", "inventory": rows}), rows


def scientific_config(method: str, target: str, seed: int, dataset_fingerprint: str) -> dict:
    if method not in {"strong_aug", "feature_plus_kl"}:
        raise ValueError(method)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "PACS",
        "outer_target": target,
        "train_domains": [domain for domain in DOMAINS if domain != target],
        "method": method,
        "trainer_method": "aug" if method == "strong_aug" else "feat_kl",
        "backbone": "resnet50",
        "pretrained": True,
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
