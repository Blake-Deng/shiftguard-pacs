#!/usr/bin/env python3
"""Run pooled cross-fold screening, corrected ResNet formal runs, and ViT controls."""
from __future__ import annotations

import csv
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DOMAINS = ["Photo", "Art_Painting", "Cartoon", "Sketch"]
SEEDS = [42, 123, 3407]
GPUS = [int(value) for value in os.environ.get("SHIFTGUARD_GPUS", "0").split(",")]
LOG_ROOT = ROOT / "logs" / "corrected"
RUN_ROOT = ROOT / "runs"

CANDIDATES = [
    {"name": "kl_005", "method": "kl", "lambda_feat": 0.0, "lambda_kl": 0.05},
    {"name": "feat_005", "method": "feat", "lambda_feat": 0.05, "lambda_kl": 0.0},
    {"name": "featkl_005", "method": "feat_kl", "lambda_feat": 0.05, "lambda_kl": 0.05},
    {"name": "featkl_010_005", "method": "feat_kl", "lambda_feat": 0.10, "lambda_kl": 0.05},
    {"name": "adaptive_005", "method": "adaptive", "lambda_feat": 0.05, "lambda_kl": 0.05},
    {"name": "adaptive_010_005", "method": "adaptive", "lambda_feat": 0.10, "lambda_kl": 0.05},
]


def task_stem(task):
    return f"{task['target']}_{task['model']}_{task['run_name']}_seed{task['seed']}"


def make_command(task):
    cmd = [
        PYTHON, str(ROOT / "shiftguard_corrected.py"),
        "--data-root", str(ROOT / "data" / "PACS"),
        "--target", task["target"],
        "--method", task["method"],
        "--run-name", task["run_name"],
        "--model", task["model"],
        "--seed", str(task["seed"]),
        "--epochs", str(task.get("epochs", 30)),
        "--batch-size", str(task.get("batch_size", 64)),
        "--workers", str(task.get("workers", 8)),
        "--lambda-feat", str(task["lambda_feat"]),
        "--lambda-kl", str(task["lambda_kl"]),
        "--temperature", str(task.get("temperature", 2.0)),
        "--gate-tau", str(task.get("gate_tau", 0.5)),
        "--warmup-epochs", str(task.get("warmup_epochs", 5)),
        "--device", "cuda:0",
        "--output", str(task["output"]),
    ]
    if task.get("skip_target_eval"):
        cmd.append("--skip-target-eval")
    if task.get("save_checkpoint"):
        cmd.append("--save-checkpoint")
    return cmd


def run_phase(name, tasks):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for task in tasks:
        task["output"].mkdir(parents=True, exist_ok=True)
        expected = task["output"] / f"{task_stem(task)}.json"
        if expected.exists():
            try:
                json.loads(expected.read_text())
                print(f"[{name}] reuse {expected}", flush=True)
                continue
            except Exception:
                pass
        pending.put(task)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            stem = task_stem(task)
            log_path = LOG_ROOT / f"{name}_{stem}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = make_command(task)
            start = time.time()
            print(f"[{name}] gpu={gpu} start {stem}", flush=True)
            with log_path.open("w") as log:
                log.write("COMMAND: " + " ".join(cmd) + "\n")
                log.flush()
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            print(f"[{name}] gpu={gpu} done code={completed.returncode} seconds={elapsed:.1f} {stem}", flush=True)
            if completed.returncode != 0:
                with lock:
                    failures.append((stem, completed.returncode, str(log_path)))
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Phase {name} failures: {failures}")


def screening_tasks():
    tasks = []
    for candidate in CANDIDATES:
        for target in DOMAINS:
            output = RUN_ROOT / "corrected_screening" / "tasks" / candidate["name"] / target
            tasks.append({
                **candidate,
                "run_name": candidate["name"],
                "target": target,
                "seed": 42,
                "model": "resnet50",
                "skip_target_eval": True,
                "save_checkpoint": False,
                "output": output,
            })
    return tasks


def select_candidate():
    rows = []
    for candidate in CANDIDATES:
        values = []
        for target in DOMAINS:
            output = RUN_ROOT / "corrected_screening" / "tasks" / candidate["name"] / target
            path = output / f"{target}_resnet50_{candidate['name']}_seed42.json"
            result = json.loads(path.read_text())
            if result["target_evaluations"] != 0 or result["target_accuracy"] is not None:
                raise RuntimeError(f"Screening target leakage detected in {path}")
            values.append(float(result["best_source_val"]))
        rows.append({
            **candidate,
            "mean_source_val": statistics.mean(values),
            "std_source_val": statistics.stdev(values),
            "per_outer_target": dict(zip(DOMAINS, values)),
        })
    rows.sort(key=lambda row: row["mean_source_val"], reverse=True)
    summary_dir = RUN_ROOT / "corrected_screening"
    (summary_dir / "screening_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (summary_dir / "screening_summary.csv").open("w", newline="") as handle:
        fields = ["name", "method", "lambda_feat", "lambda_kl", "mean_source_val", "std_source_val"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    selected = rows[0]
    (summary_dir / "selected_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("SCREENING RANKING", flush=True)
    for row in rows:
        print(f"  {row['name']}: {row['mean_source_val']:.6f} +/- {row['std_source_val']:.6f}", flush=True)
    print(f"SELECTED {selected['name']}", flush=True)
    return selected


def formal_tasks(selected):
    tasks = []
    for target in DOMAINS:
        for seed in SEEDS:
            output = RUN_ROOT / "corrected_formal" / "tasks" / target / str(seed)
            tasks.append({
                "name": selected["name"],
                "run_name": selected["name"],
                "method": selected["method"],
                "lambda_feat": selected["lambda_feat"],
                "lambda_kl": selected["lambda_kl"],
                "target": target,
                "seed": seed,
                "model": "resnet50",
                "skip_target_eval": False,
                "save_checkpoint": True,
                "output": output,
            })
    return tasks


def vit_tasks(selected):
    configs = [
        {"run_name": "strong_aug", "method": "aug", "lambda_feat": 0.0, "lambda_kl": 0.0},
        {"run_name": selected["name"], "method": selected["method"], "lambda_feat": selected["lambda_feat"], "lambda_kl": selected["lambda_kl"]},
    ]
    tasks = []
    for config in configs:
        for target in DOMAINS:
            for seed in SEEDS:
                output = RUN_ROOT / "corrected_vit" / "tasks" / config["run_name"] / target / str(seed)
                tasks.append({
                    **config,
                    "name": config["run_name"],
                    "target": target,
                    "seed": seed,
                    "model": "vit-small",
                    "skip_target_eval": False,
                    "save_checkpoint": False,
                    "output": output,
                    "batch_size": 64,
                })
    return tasks


def summarize_results(root, destination):
    records = []
    for path in root.rglob("*.json"):
        result = json.loads(path.read_text())
        if isinstance(result, dict) and result.get("target_accuracy") is not None:
            records.append(result)
    grouped = {}
    for result in records:
        key = (result["model"], result["run_name"])
        grouped.setdefault(key, []).append(result)
    summary = []
    for (model, run_name), items in sorted(grouped.items()):
        per_domain = {}
        for target in DOMAINS:
            values = [100 * x["target_accuracy"] for x in items if x["target"] == target]
            per_domain[target] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        macro_by_seed = []
        for seed in SEEDS:
            values = [100 * x["target_accuracy"] for x in items if x["seed"] == seed]
            if len(values) == len(DOMAINS):
                macro_by_seed.append(statistics.mean(values))
        summary.append({
            "model": model,
            "run_name": run_name,
            "per_domain": per_domain,
            "macro_by_seed": macro_by_seed,
            "macro_mean": statistics.mean(macro_by_seed) if macro_by_seed else None,
            "macro_std": statistics.stdev(macro_by_seed) if len(macro_by_seed) > 1 else None,
        })
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    pipeline_start = time.time()
    print("PHASE 1: source-only screening", flush=True)
    run_phase("screen", screening_tasks())
    selected = select_candidate()
    print("PHASE 2: corrected ResNet-50 formal evaluation", flush=True)
    run_phase("formal", formal_tasks(selected))
    formal_summary = summarize_results(RUN_ROOT / "corrected_formal", RUN_ROOT / "corrected_formal" / "summary.json")
    print(json.dumps(formal_summary, indent=2), flush=True)
    print("PHASE 3: ViT-S/16 Strong Aug and corrected method", flush=True)
    run_phase("vit", vit_tasks(selected))
    vit_summary = summarize_results(RUN_ROOT / "corrected_vit", RUN_ROOT / "corrected_vit" / "summary.json")
    print(json.dumps(vit_summary, indent=2), flush=True)
    final = {
        "selected": selected,
        "formal_summary": formal_summary,
        "vit_summary": vit_summary,
        "elapsed_seconds": time.time() - pipeline_start,
    }
    (RUN_ROOT / "corrected_pipeline_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
