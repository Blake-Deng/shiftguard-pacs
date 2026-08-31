#!/usr/bin/env python3
"""Recompute paper-facing summaries from published per-run JSON records."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
SEEDS = (42, 123, 3407)
EXPECTED = {"erm": 12, "mixup": 12, "coral": 12, "corrected_ablation": 48, "corrected_formal": 12, "corrected_vit": 24}

def load_records(folder: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

def summarize_group(records: list[dict]) -> list[dict]:
    grouped = {}
    for record in records:
        key = (record.get("model", "resnet50"), record.get("run_name") or record.get("method"))
        grouped.setdefault(key, []).append(record)
    output = []
    for (model, method), items in sorted(grouped.items()):
        per_domain = {}
        for domain in DOMAINS:
            domain_items = [item for item in items if item["target"] == domain and item.get("target_accuracy") is not None]
            values = [100.0 * float(item["target_accuracy"]) for item in domain_items]
            per_domain[domain] = {
                "values_by_seed": {str(item["seed"]): 100.0 * float(item["target_accuracy"]) for item in domain_items},
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }
        macro_by_seed = {}
        for seed in SEEDS:
            values = [100.0 * float(item["target_accuracy"]) for item in items if item["seed"] == seed and item.get("target_accuracy") is not None]
            if len(values) == len(DOMAINS):
                macro_by_seed[str(seed)] = statistics.mean(values)
        macro_values = list(macro_by_seed.values())
        output.append({"model": model, "method": method, "per_domain": per_domain, "macro_by_seed": macro_by_seed, "macro_mean": statistics.mean(macro_values), "macro_sample_std": statistics.stdev(macro_values)})
    return output

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reference_results")
    parser.add_argument("--output", default="reference_results/recomputed_summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    screening = load_records(root / "corrected_screening" / "runs")
    report = {
        "screening_audit": {
            "run_count": len(screening),
            "all_target_evaluations_zero": all(record.get("target_evaluations") == 0 for record in screening),
            "all_target_accuracy_null": all(record.get("target_accuracy") is None for record in screening),
        },
        "groups": {},
    }
    if len(screening) != 24:
        raise RuntimeError(f"Expected 24 screening records, found {len(screening)}")
    for group, expected in EXPECTED.items():
        records = load_records(root / group / "runs")
        if len(records) != expected:
            raise RuntimeError(f"{group}: expected {expected} JSON files, found {len(records)}")
        report["groups"][group] = summarize_group(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
