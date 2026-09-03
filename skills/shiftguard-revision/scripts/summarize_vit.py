#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from revision_common import DOMAINS, FORMAL_SEEDS, project_root

METHODS = ("strong_aug", "feature_plus_kl")
T_CRITICAL_DF4 = 2.7764451051977987


def result_path(root: Path, method: str, domain: str, seed: int) -> Path:
    return root / "runs" / "revision" / "vit_direct" / method / domain / str(seed) / f"{domain}_vit-small_{method}_seed{seed}.json"


def load_accuracy(path: Path) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("target_evaluations") != 1 or result.get("target_accuracy") is None:
        raise RuntimeError(f"invalid formal result: {path}")
    if result.get("config", {}).get("preprocessing") != "timm_vit_standard":
        raise RuntimeError(f"preprocessing mismatch: {path}")
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
    summary, macros = {}, {}
    for method in METHODS:
        per_domain = {}
        for domain in DOMAINS:
            samples = [values[method][seed][domain] for seed in FORMAL_SEEDS]
            per_domain[domain] = {"values": samples, "mean": statistics.mean(samples), "sample_sd": statistics.stdev(samples)}
        method_macros = [statistics.mean(values[method][seed].values()) for seed in FORMAL_SEEDS]
        macros[method] = method_macros
        summary[method] = {
            "per_domain": per_domain,
            "macro_by_seed": dict(zip(FORMAL_SEEDS, method_macros)),
            "macro_mean": statistics.mean(method_macros),
            "macro_sample_sd": statistics.stdev(method_macros),
        }
    deltas = [feature - aug for aug, feature in zip(macros["strong_aug"], macros["feature_plus_kl"])]
    mean_delta = statistics.mean(deltas)
    sd_delta = statistics.stdev(deltas)
    half_width = T_CRITICAL_DF4 * sd_delta / math.sqrt(5)
    payload = {
        "protocol_version": "shiftguard-vit-direct-v1.0",
        "backbone": "timm:vit_small_patch16_224.augreg_in21k_ft_in1k",
        "preprocessing": "timm_vit_standard",
        "formal_seeds": FORMAL_SEEDS,
        "methods": summary,
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
    output = destination / "vit_direct_five_seed_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    with (destination / "vit_direct_five_seed_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *DOMAINS, "macro"])
        for method in METHODS:
            cells = [f"{summary[method]['per_domain'][domain]['mean']:.2f} +/- {summary[method]['per_domain'][domain]['sample_sd']:.2f}" for domain in DOMAINS]
            cells.append(f"{summary[method]['macro_mean']:.2f} +/- {summary[method]['macro_sample_sd']:.2f}")
            writer.writerow([method, *cells])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
