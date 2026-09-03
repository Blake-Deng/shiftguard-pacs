#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from revision_common import project_root


def validate(row: dict, result: dict) -> None:
    checks = {
        "target": row["outer_target"],
        "method": row["trainer_method"],
        "run_name": row["method"],
        "model": row["backbone"],
        "seed": int(row["seed"]),
        "target_evaluations": 1,
    }
    for key, expected in checks.items():
        if result.get(key) != expected:
            raise RuntimeError(f"{key}: expected {expected!r}, found {result.get(key)!r}")
    accuracy = result.get("target_accuracy")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise RuntimeError(f"invalid target_accuracy: {accuracy!r}")
    config = result.get("config", {})
    numeric = {
        "lambda_feat": float(row["lambda_f"]),
        "lambda_kl": float(row["lambda_k"]),
        "temperature": float(row["temperature"]),
        "warmup_epochs": 5,
    }
    for key, expected in numeric.items():
        if abs(float(config.get(key, float("nan"))) - expected) > 1e-12:
            raise RuntimeError(f"config {key}: expected {expected}, found {config.get(key)!r}")
    audit = result.get("audit")
    if audit is not None and audit.get("config_hash") != row["config_hash"]:
        raise RuntimeError("audit config_hash does not match manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--manifest", default="revision/revision_manifest.csv")
    args = parser.parse_args()
    root = project_root()
    with (root / args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] == args.experiment_id]
    if len(rows) != 1:
        raise RuntimeError(f"expected one manifest row, found {len(rows)}")
    row = rows[0]
    result = json.loads((root / row["result_path"]).read_text(encoding="utf-8"))
    validate(row, result)
    print(f"PASS {args.experiment_id} target_accuracy={100 * result['target_accuracy']:.4f}%")


if __name__ == "__main__":
    main()
