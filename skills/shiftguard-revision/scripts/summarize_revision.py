#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from revision_common import DOMAINS, FORMAL_SEEDS, project_root

T_CRITICAL_DF4 = 2.7764451051977987
METHODS = ("strong_aug", "feature_plus_kl")


def result_path(root: Path, method: str, domain: str, seed: int) -> Path:
    stem = f"{domain}_resnet50_{method}_seed{seed}.json"
    if seed in (42, 123, 3407):
        return root / "runs" / "corrected_ablation" / "tasks" / method / domain / str(seed) / stem
    return root / "runs" / "revision" / "pacs5" / method / domain / str(seed) / stem


def load_accuracy(path: Path) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("target_evaluations") != 1 or result.get("target_accuracy") is None:
        raise RuntimeError(f"invalid formal result: {path}")
    return 100.0 * float(result["target_accuracy"])


def main() -> None:
    root = project_root()
    values = {
        method: {
            seed: {domain: load_accuracy(result_path(root, method, domain, seed)) for domain in DOMAINS}
            for seed in FORMAL_SEEDS
        }
        for method in METHODS
    }
    summary = {}
    macro_by_method = {}
    for method in METHODS:
        per_domain = {}
        for domain in DOMAINS:
            samples = [values[method][seed][domain] for seed in FORMAL_SEEDS]
            per_domain[domain] = {
                "values": samples,
                "mean": statistics.mean(samples),
                "sample_sd": statistics.stdev(samples),
            }
        macros = [statistics.mean(values[method][seed].values()) for seed in FORMAL_SEEDS]
        macro_by_method[method] = macros
        summary[method] = {
            "per_domain": per_domain,
            "macro_by_seed": dict(zip(FORMAL_SEEDS, macros)),
            "macro_mean": statistics.mean(macros),
            "macro_sample_sd": statistics.stdev(macros),
        }
    deltas = [b - a for a, b in zip(macro_by_method["strong_aug"], macro_by_method["feature_plus_kl"])]
    delta_mean = statistics.mean(deltas)
    delta_sd = statistics.stdev(deltas)
    half_width = T_CRITICAL_DF4 * delta_sd / math.sqrt(len(deltas))
    paired = {
        "delta_by_seed": dict(zip(FORMAL_SEEDS, deltas)),
        "mean_delta": delta_mean,
        "sample_sd_delta": delta_sd,
        "ci95": [delta_mean - half_width, delta_mean + half_width],
        "ci_contains_zero": delta_mean - half_width <= 0 <= delta_mean + half_width,
    }
    payload = {
        "unit_of_replication": "full training replicate seed; domains averaged within seed",
        "formal_seeds": FORMAL_SEEDS,
        "methods": summary,
        "paired_feature_plus_kl_minus_strong_aug": paired,
    }
    output_dir = root / "revision" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pacs_five_seed_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)

    csv_path = output_dir / "pacs_five_seed_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *DOMAINS, "macro"])
        for method in METHODS:
            cells = [f"{summary[method]['per_domain'][d]['mean']:.2f} +/- {summary[method]['per_domain'][d]['sample_sd']:.2f}" for d in DOMAINS]
            cells.append(f"{summary[method]['macro_mean']:.2f} +/- {summary[method]['macro_sample_sd']:.2f}")
            writer.writerow([method, *cells])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
