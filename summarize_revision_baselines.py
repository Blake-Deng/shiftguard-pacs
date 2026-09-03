#!/usr/bin/env python3
"""Aggregate revision runs using the training seed as the replicate."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revision" / "summaries"
SEEDS = [42, 123, 3407, 2026, 2027]
DOMAINS = {
    "pacs": ["Photo", "Art_Painting", "Cartoon", "Sketch"],
    "vlcs": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "officehome": ["Art", "Clipart", "Product", "Real_World"],
}


def load_manifest(path: Path) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "complete":
                raise RuntimeError(f"incomplete run: {row['experiment_id']}")
            payload = json.loads((ROOT / row["result_path"]).read_text(encoding="utf-8"))
            if payload.get("target_evaluations") != 1:
                raise RuntimeError(f"invalid target audit: {row['experiment_id']}")
            if row["outer_target"] in payload.get("source_domains", []):
                raise RuntimeError(f"target leakage: {row['experiment_id']}")
            key = (row["dataset"], row["outer_target"], int(row["seed"]))
            if key in values:
                raise RuntimeError(f"duplicate result: {key}")
            values[key] = 100.0 * float(payload["target_accuracy"])
    return values


def summarize_values(values: dict[tuple[str, str, int], float], dataset: str) -> dict:
    per_domain = {}
    for domain in DOMAINS[dataset]:
        scores = [values[(dataset, domain, seed)] for seed in SEEDS]
        per_domain[domain] = {
            "values": scores,
            "mean": statistics.mean(scores),
            "sample_sd": statistics.stdev(scores),
        }
    macro_by_seed = {
        str(seed): statistics.mean(values[(dataset, domain, seed)] for domain in DOMAINS[dataset])
        for seed in SEEDS
    }
    macros = list(macro_by_seed.values())
    return {
        "per_domain": per_domain,
        "macro_by_seed": macro_by_seed,
        "macro_mean": statistics.mean(macros),
        "macro_sample_sd": statistics.stdev(macros),
    }


def main() -> None:
    cross = json.loads((OUT / "cross_dataset_five_seed_summary.json").read_text(encoding="utf-8"))
    pacs = json.loads((OUT / "pacs_five_seed_summary.json").read_text(encoding="utf-8"))
    methods = {
        "mixstyle": load_manifest(ROOT / "revision" / "mixstyle_manifest.csv"),
        "swad_epoch": load_manifest(ROOT / "revision" / "swad_manifest.csv"),
    }

    summary = {
        "unit_of_replication": "training seed; target domains are averaged within each seed",
        "formal_seeds": SEEDS,
        "datasets": {},
    }
    for dataset in DOMAINS:
        dataset_methods = {}
        if dataset == "pacs":
            dataset_methods.update(pacs["methods"])
        else:
            dataset_methods.update(cross["datasets"][dataset]["methods"])
        for method, values in methods.items():
            dataset_methods[method] = summarize_values(values, dataset)
        summary["datasets"][dataset] = {"methods": dataset_methods}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all_methods_five_seed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for dataset, block in summary["datasets"].items():
        for method, result in block["methods"].items():
            rows.append({
                "dataset": dataset,
                "method": method,
                "macro_mean": result["macro_mean"],
                "macro_sample_sd": result["macro_sample_sd"],
                "n_seeds": 5,
            })
    with (OUT / "all_methods_macro_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['dataset']:10s} {row['method']:16s} "
              f"{row['macro_mean']:.2f} +/- {row['macro_sample_sd']:.2f}")


if __name__ == "__main__":
    main()
