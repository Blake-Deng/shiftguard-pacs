#!/usr/bin/env python3
"""Summarize the compact PACS OAT sensitivity analysis."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
GRIDS = {
    "lambda_f": ("0.05", "0.10", "0.20"),
    "lambda_k": ("0.025", "0.05", "0.10"),
    "temperature": ("1", "2", "4"),
    "augmentation_M": ("5", "9", "13"),
}
DEFAULTS = {"lambda_f": "0.10", "lambda_k": "0.05", "temperature": "2", "augmentation_M": "9"}


def base_result(target: str) -> dict:
    path = ROOT / "runs" / "corrected_ablation" / "tasks" / "feature_plus_kl" / target / "42" / f"{target}_resnet50_feature_plus_kl_seed42.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {"lambda_feat": 0.10, "lambda_kl": 0.05, "temperature": 2.0}
    if result.get("target_evaluations") != 1 or result.get("target_accuracy") is None:
        raise RuntimeError(f"invalid reused base result: {path}")
    for key, value in expected.items():
        if abs(float(result.get("config", {}).get(key, float("nan"))) - value) > 1e-12:
            raise RuntimeError(f"base config mismatch for {key}: {path}")
    return result


def main() -> None:
    with (ROOT / "revision" / "sensitivity_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32 or any(row["status"] != "complete" for row in rows):
        raise RuntimeError("sensitivity manifest is not complete")
    indexed = {(row["sweep"], row["factor_value"], row["outer_target"]): row for row in rows}
    base = {target: base_result(target) for target in DOMAINS}
    records, nested = [], {}
    for sweep, values in GRIDS.items():
        nested[sweep] = []
        for value in values:
            target_acc, source_val = {}, {}
            for target in DOMAINS:
                if value == DEFAULTS[sweep]:
                    result = base[target]
                else:
                    row = indexed[(sweep, value, target)]
                    result = json.loads((ROOT / row["result_path"]).read_text(encoding="utf-8"))
                    if result.get("audit", {}).get("config_hash") != row["config_hash"]:
                        raise RuntimeError(f"audit mismatch: {row['experiment_id']}")
                target_acc[target] = 100.0 * float(result["target_accuracy"])
                source_val[target] = 100.0 * float(result["best_source_val"])
            record = {
                "sweep": sweep, "value": value,
                "source_validation_macro": statistics.mean(source_val.values()),
                "target_macro": statistics.mean(target_acc.values()),
                "target_by_domain": target_acc, "source_validation_by_fold": source_val,
                "seed": 42, "reused_default": value == DEFAULTS[sweep],
            }
            nested[sweep].append(record); records.append(record)
    payload = {
        "protocol_version": "shiftguard-compact-sensitivity-v1.0",
        "scope": "descriptive single-seed OAT sensitivity; not used for configuration selection",
        "seed": 42, "domains": DOMAINS, "default_configuration": {
            "lambda_f": 0.10, "lambda_k": 0.05, "temperature": 2.0,
            "augmentation_N": 2, "augmentation_M": 9,
        }, "sweeps": nested,
    }
    out = ROOT / "revision" / "summaries"; out.mkdir(parents=True, exist_ok=True)
    (out / "compact_sensitivity_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "compact_sensitivity_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sweep", "value", "seed", "source_validation_macro", *DOMAINS, "target_macro", "reused_default"])
        for record in records:
            writer.writerow([record["sweep"], record["value"], 42, record["source_validation_macro"], *[record["target_by_domain"][d] for d in DOMAINS], record["target_macro"], record["reused_default"]])
    print(json.dumps({"rows": len(records), "output": str(out / "compact_sensitivity_table.csv")}, indent=2))


if __name__ == "__main__":
    main()
