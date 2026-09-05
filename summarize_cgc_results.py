#!/usr/bin/env python3
"""Build machine-readable CGC tables, paired statistics, and diagnostics."""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

from shiftguard_corrected import build_model
from run_cgc_complete_queue import DATASETS, FIELDS, ROOT, SEEDS

SUMMARY_DIR = ROOT / "revision/cgc_v2_summaries"
T_CRITICAL_DF4 = 2.7764451051977987


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest() -> list[dict]:
    path = ROOT / "revision/cgc_v2_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 232:
        raise RuntimeError(f"expected 232 manifest rows, found {len(rows)}")
    incomplete = [row["experiment_id"] for row in rows if row["status"] != "complete"]
    if incomplete:
        raise RuntimeError(f"cannot summarize {len(incomplete)} incomplete jobs")
    return rows


def new_result(rows: list[dict], group: str, dataset: str, model: str, method: str, target: str, seed: int) -> dict:
    matches = [
        row for row in rows
        if row["experiment_group"] == group
        and row["dataset"] == dataset
        and row["model"] == model
        and row["method"] == method
        and row["target"] == target
        and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one result for {(group, dataset, model, method, target, seed)}, found {len(matches)}")
    return read_json(ROOT / matches[0]["result_path"])


def existing_result(dataset: str, model: str, method: str, target: str, seed: int) -> dict:
    if model == "vit-small":
        path = ROOT / "runs/revision/vit_direct" / method / target / str(seed) / f"{target}_vit-small_{method}_seed{seed}.json"
    elif dataset == "pacs":
        if seed in (42, 123, 3407):
            path = ROOT / "runs/corrected_ablation/tasks" / method / target / str(seed) / f"{target}_resnet50_{method}_seed{seed}.json"
        else:
            path = ROOT / "runs/revision/pacs5" / method / target / str(seed) / f"{target}_resnet50_{method}_seed{seed}.json"
    else:
        path = ROOT / "runs/revision/cross_dataset" / dataset / method / target / str(seed) / f"{dataset}_{target}_resnet50_{method}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def accuracy(rows: list[dict], dataset: str, model: str, method: str, target: str, seed: int) -> float:
    if method in {"strong_aug", "feature_plus_kl"}:
        result = existing_result(dataset, model, method, target, seed)
    elif method == "cgc":
        result = new_result(rows, "formal_cgc", dataset, model, method, target, seed)
    elif method == "mean_teacher":
        result = new_result(rows, "teacher", dataset, model, method, target, seed)
    else:
        raise ValueError(method)
    if result.get("target_evaluations") != 1:
        raise RuntimeError("formal result does not have exactly one target evaluation")
    return 100.0 * float(result["target_accuracy"])


def sample_stats(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def method_summary(rows: list[dict], dataset: str, model: str, method: str) -> dict:
    per_domain = {
        target: sample_stats([accuracy(rows, dataset, model, method, target, seed) for seed in SEEDS])
        for target in DATASETS[dataset]
    }
    macro_by_seed = {
        str(seed): statistics.fmean([
            accuracy(rows, dataset, model, method, target, seed) for target in DATASETS[dataset]
        ])
        for seed in SEEDS
    }
    return {
        "per_domain": per_domain,
        "macro_by_seed": macro_by_seed,
        "macro": sample_stats(list(macro_by_seed.values())),
    }


def paired(left: dict, right: dict) -> dict:
    deltas = [
        float(left["macro_by_seed"][str(seed)]) - float(right["macro_by_seed"][str(seed)])
        for seed in SEEDS
    ]
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas)
    half_width = T_CRITICAL_DF4 * sd / math.sqrt(len(deltas))
    return {
        "deltas_by_seed": {str(seed): value for seed, value in zip(SEEDS, deltas)},
        "mean": mean,
        "sample_sd": sd,
        "ci95": [mean - half_width, mean + half_width],
        "ci_contains_zero": mean - half_width <= 0 <= mean + half_width,
    }


def baseline_macros() -> dict:
    output: dict[str, dict[str, dict]] = {}
    path = ROOT / "revision/summaries/all_methods_macro_table.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.setdefault(row["dataset"], {})[row["method"]] = {
                "mean": float(row["macro_mean"]),
                "sample_sd": float(row["macro_sample_sd"]),
                "n": int(row["n_seeds"]),
            }
    return output


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else None


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def mechanism(rows: list[dict]) -> dict:
    observations = []
    for dataset in DATASETS:
        for target in DATASETS[dataset]:
            for seed in SEEDS:
                result = new_result(rows, "formal_cgc", dataset, "resnet50", "cgc", target, seed)
                feature_conflict = statistics.fmean(epoch["feature_conflict_rate"] for epoch in result["history"])
                kl_conflict = statistics.fmean(epoch["kl_conflict_rate"] for epoch in result["history"])
                strong = accuracy(rows, dataset, "resnet50", "strong_aug", target, seed)
                feature_kl = accuracy(rows, dataset, "resnet50", "feature_plus_kl", target, seed)
                cgc = 100.0 * float(result["target_accuracy"])
                observations.append({
                    "dataset": dataset,
                    "target": target,
                    "seed": seed,
                    "feature_conflict_rate": feature_conflict,
                    "kl_conflict_rate": kl_conflict,
                    "feature_kl_minus_strong_aug": feature_kl - strong,
                    "cgc_minus_feature_kl": cgc - feature_kl,
                })

    def correlations(items: list[dict]) -> dict:
        return {
            "n": len(items),
            "feature_conflict_vs_featurekl_minus_aug": spearman(
                [item["feature_conflict_rate"] for item in items],
                [item["feature_kl_minus_strong_aug"] for item in items],
            ),
            "feature_conflict_vs_cgc_minus_featurekl": spearman(
                [item["feature_conflict_rate"] for item in items],
                [item["cgc_minus_feature_kl"] for item in items],
            ),
            "kl_conflict_vs_featurekl_minus_aug": spearman(
                [item["kl_conflict_rate"] for item in items],
                [item["feature_kl_minus_strong_aug"] for item in items],
            ),
            "kl_conflict_vs_cgc_minus_featurekl": spearman(
                [item["kl_conflict_rate"] for item in items],
                [item["cgc_minus_feature_kl"] for item in items],
            ),
        }

    return {
        "scope_note": "Descriptive correlations; dataset-target-seed observations are not claimed as independent causal replicates.",
        "overall": correlations(observations),
        "per_dataset": {
            dataset: correlations([item for item in observations if item["dataset"] == dataset])
            for dataset in DATASETS
        },
        "observations": observations,
    }


def negative_transfer(rows: list[dict], method: str, datasets: tuple[str, ...]) -> dict:
    output = {}
    all_flags = []
    for dataset in datasets:
        flags = []
        for target in DATASETS[dataset]:
            for seed in SEEDS:
                strong = accuracy(rows, dataset, "resnet50", "strong_aug", target, seed)
                candidate = accuracy(rows, dataset, "resnet50", method, target, seed)
                flags.append(candidate < strong)
        all_flags.extend(flags)
        output[dataset] = {"negative": sum(flags), "total": len(flags), "rate": sum(flags) / len(flags)}
    output["overall"] = {
        "negative": sum(all_flags), "total": len(all_flags), "rate": sum(all_flags) / len(all_flags)
    }
    return output


def ablations(rows: list[dict]) -> list[dict]:
    output = []
    for method in ("cgc_feature_gate", "cgc_kl_gate", "cgc_combined", "cgc_random"):
        results = [new_result(rows, "ablation", "pacs", "resnet50", method, target, 42) for target in DATASETS["pacs"]]
        accuracies = [100.0 * float(result["target_accuracy"]) for result in results]
        strong = [accuracy(rows, "pacs", "resnet50", "strong_aug", target, 42) for target in DATASETS["pacs"]]
        output.append({
            "method": method,
            "scope": "PACS seed 42 descriptive",
            "per_domain": dict(zip(DATASETS["pacs"], accuracies)),
            "macro": statistics.fmean(accuracies),
            "negative_transfer_rate": sum(a < b for a, b in zip(accuracies, strong)) / len(strong),
            "feature_keep_rate": statistics.fmean(
                epoch["feature_keep_rate"] for result in results for epoch in result["history"]
            ),
            "kl_keep_rate": statistics.fmean(
                epoch["kl_keep_rate"] for result in results for epoch in result["history"]
            ),
        })
    return output


def nested_selection(rows: list[dict]) -> dict:
    candidate_order = ("strong_aug", "feature_plus_kl", "cgc")
    outer_rows = []
    selected_by_target = {}
    for outer_target in DATASETS["pacs"]:
        candidate_scores = {}
        for method in candidate_order:
            matches = [
                row for row in rows
                if row["experiment_group"] == "nested"
                and row["target"] == outer_target
                and row["method"] == method
            ]
            if len(matches) != 9:
                raise RuntimeError(f"expected nine nested scores for {outer_target}/{method}")
            candidate_scores[method] = statistics.fmean(
                float(read_json(ROOT / row["result_path"])["best_source_val"]) * 100.0 for row in matches
            )
        selected = max(candidate_order, key=lambda method: candidate_scores[method])
        selected_by_target[outer_target] = selected
        target_values = [accuracy(rows, "pacs", "resnet50", selected, outer_target, seed) for seed in SEEDS]
        outer_rows.append({
            "outer_target": outer_target,
            "candidate_inner_scores": candidate_scores,
            "selected_method": selected,
            "formal_target": sample_stats(target_values),
        })
    macro_by_seed = {
        str(seed): statistics.fmean([
            accuracy(rows, "pacs", "resnet50", selected_by_target[target], target, seed)
            for target in DATASETS["pacs"]
        ])
        for seed in SEEDS
    }
    return {
        "selection_rule": "Highest mean inner-domain validation accuracy over 3 inner folds x 3 screening seeds; candidate order only resolves exact ties.",
        "outer_targets": outer_rows,
        "nested_macro": sample_stats(list(macro_by_seed.values())),
        "macro_by_seed": macro_by_seed,
    }


def benchmark_latency(checkpoint: Path) -> tuple[int, float]:
    model = build_model("resnet50", 7, False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model = model.cuda().eval()
    sample = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(20):
            model(sample)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            model(sample)
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0 / 100.0
    del model, sample, payload
    torch.cuda.empty_cache()
    return parameters, latency_ms


def efficiency(rows: list[dict]) -> list[dict]:
    output = []
    for method in ("strong_aug", "feature_plus_kl", "cgc"):
        group = "formal_cgc" if method == "cgc" else "efficiency"
        results = [new_result(rows, group, "pacs", "resnet50", method, target, 42) for target in DATASETS["pacs"]]
        manifest_match = next(
            row for row in rows
            if row["experiment_group"] == group and row["dataset"] == "pacs"
            and row["method"] == method and row["target"] == "Photo" and int(row["seed"]) == 42
        )
        checkpoint = (ROOT / manifest_match["result_path"]).with_suffix(".pt")
        parameters, latency_ms = benchmark_latency(checkpoint)
        output.append({
            "method": method,
            "train_seconds_per_epoch_mean_over_targets": statistics.fmean(
                result["elapsed_seconds"] / 30.0 for result in results
            ),
            "total_train_minutes_mean_over_targets": statistics.fmean(
                result["elapsed_seconds"] / 60.0 for result in results
            ),
            "peak_gpu_memory_gib_max_over_targets": max(
                result["peak_gpu_memory_bytes"] for result in results
            ) / (1024 ** 3),
            "parameters": parameters,
            "batch1_inference_latency_ms": latency_ms,
            "inference_protocol": "100 FP16 forward passes after 20 warmups on one RTX 5090; batch size 1.",
        })
    return output


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_manifest()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for dataset in DATASETS:
        summaries[dataset] = {
            method: method_summary(rows, dataset, "resnet50", method)
            for method in ("strong_aug", "feature_plus_kl", "cgc")
        }
        if dataset == "pacs":
            summaries[dataset]["mean_teacher"] = method_summary(rows, dataset, "resnet50", "mean_teacher")
        summaries[dataset]["paired_cgc_minus_strong_aug"] = paired(
            summaries[dataset]["cgc"], summaries[dataset]["strong_aug"]
        )
        summaries[dataset]["paired_cgc_minus_feature_plus_kl"] = paired(
            summaries[dataset]["cgc"], summaries[dataset]["feature_plus_kl"]
        )

    vit = {
        method: method_summary(rows, "pacs", "vit-small", method)
        for method in ("strong_aug", "feature_plus_kl", "cgc")
    }
    vit["paired_cgc_minus_strong_aug"] = paired(vit["cgc"], vit["strong_aug"])
    vit["paired_cgc_minus_feature_plus_kl"] = paired(vit["cgc"], vit["feature_plus_kl"])

    ntr = {
        "feature_plus_kl": negative_transfer(rows, "feature_plus_kl", tuple(DATASETS)),
        "cgc": negative_transfer(rows, "cgc", tuple(DATASETS)),
        "mean_teacher": negative_transfer(rows, "mean_teacher", ("pacs",)),
    }
    payload = {
        "protocol_version": "cgc-v2-zero-threshold-2026-09-04",
        "formal_seeds": SEEDS,
        "unit_of_replication": "training seed; domains are not independent replicates",
        "resnet50": summaries,
        "vit_small_patch16_224": vit,
        "negative_transfer_rate": ntr,
        "mechanism": mechanism(rows),
        "ablations": ablations(rows),
        "strict_nested_pacs": nested_selection(rows),
        "efficiency": efficiency(rows),
    }
    (SUMMARY_DIR / "complete_cgc_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    baselines = baseline_macros()
    main_rows = []
    for method in ("strong_aug", "feature_plus_kl", "cgc", "mixstyle", "swad_epoch", "mean_teacher"):
        row = {"method": method}
        for dataset in DATASETS:
            if method in summaries[dataset]:
                stats = summaries[dataset][method]["macro"]
                row[dataset] = f"{stats['mean']:.2f} +/- {stats['sample_sd']:.2f}"
            elif method in baselines.get(dataset, {}):
                stats = baselines[dataset][method]
                row[dataset] = f"{stats['mean']:.2f} +/- {stats['sample_sd']:.2f}"
            else:
                row[dataset] = "not run"
        main_rows.append(row)
    write_csv(SUMMARY_DIR / "table_main_multibench.csv", ["method", *DATASETS], main_rows)

    pacs_rows = []
    for method in ("strong_aug", "feature_plus_kl", "cgc", "mean_teacher"):
        summary = summaries["pacs"][method]
        row = {"method": method}
        for target in DATASETS["pacs"]:
            stats = summary["per_domain"][target]
            row[target] = f"{stats['mean']:.2f} +/- {stats['sample_sd']:.2f}"
        row["macro"] = f"{summary['macro']['mean']:.2f} +/- {summary['macro']['sample_sd']:.2f}"
        pacs_rows.append(row)
    write_csv(
        SUMMARY_DIR / "table_pacs_detail.csv",
        ["method", *DATASETS["pacs"], "macro"],
        pacs_rows,
    )
    write_csv(
        SUMMARY_DIR / "table_efficiency.csv",
        list(payload["efficiency"][0]),
        payload["efficiency"],
    )
    print(json.dumps({
        "summary": str(SUMMARY_DIR / "complete_cgc_summary.json"),
        "main_table": str(SUMMARY_DIR / "table_main_multibench.csv"),
        "pacs_table": str(SUMMARY_DIR / "table_pacs_detail.csv"),
        "efficiency_table": str(SUMMARY_DIR / "table_efficiency.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
