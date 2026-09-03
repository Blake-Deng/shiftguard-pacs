#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from create_cross_manifest import DATASETS, METHODS, PROTOCOL
from revision_common import FORMAL_SEEDS, project_root

T_CRITICAL_DF4 = 2.7764451051977987


def result_path(root: Path, dataset: str, method: str, target: str, seed: int) -> Path:
    stem = f"{dataset}_{target}_resnet50_{method}_seed{seed}.json"
    return root / "runs" / "revision" / "cross_dataset" / dataset / method / target / str(seed) / stem


def load_accuracy(path: Path) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("target_evaluations") != 1 or result.get("target_accuracy") is None:
        raise RuntimeError(f"invalid formal result: {path}")
    return 100.0 * float(result["target_accuracy"])


def main() -> None:
    root = project_root()
    payload = {"protocol_version": PROTOCOL, "datasets": {}}
    macro_csv = []
    for dataset, spec in DATASETS.items():
        values = {
            method: {
                seed: {target: load_accuracy(result_path(root, dataset, method, target, seed)) for target in spec["targets"]}
                for seed in FORMAL_SEEDS
            }
            for method in METHODS
        }
        method_summary, macros = {}, {}
        for method in METHODS:
            per_domain = {}
            for target in spec["targets"]:
                samples = [values[method][seed][target] for seed in FORMAL_SEEDS]
                per_domain[target] = {"values": samples, "mean": statistics.mean(samples), "sample_sd": statistics.stdev(samples)}
            method_macros = [statistics.mean(values[method][seed].values()) for seed in FORMAL_SEEDS]
            macros[method] = method_macros
            method_summary[method] = {
                "per_domain": per_domain,
                "macro_by_seed": dict(zip(FORMAL_SEEDS, method_macros)),
                "macro_mean": statistics.mean(method_macros),
                "macro_sample_sd": statistics.stdev(method_macros),
            }
            macro_csv.append([dataset, method, statistics.mean(method_macros), statistics.stdev(method_macros)])
        deltas = [feature - aug for aug, feature in zip(macros["strong_aug"], macros["feature_plus_kl"])]
        mean_delta, sd_delta = statistics.mean(deltas), statistics.stdev(deltas)
        half_width = T_CRITICAL_DF4 * sd_delta / math.sqrt(5)
        payload["datasets"][dataset] = {
            "methods": method_summary,
            "paired_feature_plus_kl_minus_strong_aug": {
                "delta_by_seed": dict(zip(FORMAL_SEEDS, deltas)),
                "mean_delta": mean_delta,
                "sample_sd_delta": sd_delta,
                "ci95": [mean_delta - half_width, mean_delta + half_width],
                "ci_contains_zero": mean_delta - half_width <= 0 <= mean_delta + half_width,
            },
        }
    destination = root / "revision" / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "cross_dataset_five_seed_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    with (destination / "cross_dataset_macro_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "method", "macro_mean", "macro_sample_sd"])
        writer.writerows(macro_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
