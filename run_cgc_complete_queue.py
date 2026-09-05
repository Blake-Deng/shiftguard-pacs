#!/usr/bin/env python3
"""Run the CGC revision matrix with a restartable three-GPU task queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torchvision

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 3407, 2026, 2027)
SCREENING_SEEDS = (42, 123, 3407)
DATASETS = {
    "pacs": ("Photo", "Art_Painting", "Cartoon", "Sketch"),
    "vlcs": ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
    "officehome": ("Art", "Clipart", "Product", "Real_World"),
}
DATA_ROOTS = {
    "pacs": "data/PACS",
    "vlcs": "data/VLCS",
    "officehome": "data/OfficeHome",
}
FIELDS = (
    "experiment_id", "priority", "experiment_group", "dataset", "target", "method",
    "model", "seed", "selection_protocol", "train_domains", "validation_domain",
    "epochs", "batch_size", "result_path", "diagnostics_path", "save_checkpoint",
    "status", "gpu", "attempts", "exit_code", "error",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(*parts) -> str:
    return "_".join(str(part).replace("-", "_") for part in parts)


def result_filename(row: dict) -> str:
    run_name = row["method"] if row["experiment_group"] not in {"nested", "efficiency"} else row["experiment_id"]
    return f"{row['dataset']}_{row['target']}_{row['model']}_{run_name}_seed{row['seed']}.json"


def add_task(rows: list[dict], **values) -> None:
    row = {field: "" for field in FIELDS}
    row.update(values)
    row["status"] = "pending"
    row["attempts"] = "0"
    row["exit_code"] = ""
    row["error"] = ""
    row["epochs"] = str(values.get("epochs", 30))
    row["batch_size"] = str(values.get("batch_size", 64))
    row["seed"] = str(row["seed"])
    row["priority"] = str(row["priority"])
    row["save_checkpoint"] = "1" if row.get("save_checkpoint") else "0"
    output_dir = Path("runs/cgc_v2") / row["experiment_group"] / row["dataset"] / row["model"] / row["method"] / row["target"] / row["seed"]
    row["result_path"] = str(output_dir / result_filename(row))
    row["diagnostics_path"] = str(Path("results/cgc_diagnostics") / row["experiment_group"] / row["dataset"] / row["model"] / row["method"] / row["target"] / row["seed"])
    rows.append(row)


def build_manifest() -> list[dict]:
    rows: list[dict] = []
    for dataset in ("pacs", "vlcs", "officehome"):
        for target in DATASETS[dataset]:
            for seed in SEEDS:
                add_task(
                    rows,
                    experiment_id=slug("formal", dataset, "resnet50", "cgc", target, seed),
                    priority=10,
                    experiment_group="formal_cgc",
                    dataset=dataset,
                    target=target,
                    method="cgc",
                    model="resnet50",
                    seed=seed,
                    selection_protocol="fixed_source_validation",
                    train_domains="",
                    validation_domain="",
                    save_checkpoint=True,
                )
    for target in DATASETS["pacs"]:
        for seed in SEEDS:
            add_task(
                rows,
                experiment_id=slug("formal", "pacs", "vit_small", "cgc", target, seed),
                priority=11,
                experiment_group="formal_cgc",
                dataset="pacs",
                target=target,
                method="cgc",
                model="vit-small",
                seed=seed,
                selection_protocol="fixed_source_validation",
                train_domains="",
                validation_domain="",
                save_checkpoint=True,
            )
    for target in DATASETS["pacs"]:
        for seed in SEEDS:
            add_task(
                rows,
                experiment_id=slug("formal", "pacs", "resnet50", "mean_teacher", target, seed),
                priority=20,
                experiment_group="teacher",
                dataset="pacs",
                target=target,
                method="mean_teacher",
                model="resnet50",
                seed=seed,
                selection_protocol="fixed_source_validation",
                train_domains="",
                validation_domain="",
                save_checkpoint=True,
            )
    for method in ("cgc_feature_gate", "cgc_kl_gate", "cgc_combined", "cgc_random"):
        for target in DATASETS["pacs"]:
            add_task(
                rows,
                experiment_id=slug("ablation", method, target, 42),
                priority=30,
                experiment_group="ablation",
                dataset="pacs",
                target=target,
                method=method,
                model="resnet50",
                seed=42,
                selection_protocol="fixed_source_validation",
                train_domains="",
                validation_domain="",
                save_checkpoint=False,
            )
    for method in ("strong_aug", "feature_plus_kl"):
        for target in DATASETS["pacs"]:
            add_task(
                rows,
                experiment_id=slug("efficiency", method, target, 42),
                priority=31,
                experiment_group="efficiency",
                dataset="pacs",
                target=target,
                method=method,
                model="resnet50",
                seed=42,
                selection_protocol="fixed_source_validation",
                train_domains="",
                validation_domain="",
                save_checkpoint=True,
            )
    pacs_domains = DATASETS["pacs"]
    for outer_target in pacs_domains:
        outer_sources = [domain for domain in pacs_domains if domain != outer_target]
        for validation_domain in outer_sources:
            train_domains = [domain for domain in outer_sources if domain != validation_domain]
            for method in ("strong_aug", "feature_plus_kl", "cgc"):
                for seed in SCREENING_SEEDS:
                    experiment_id = slug("nested", outer_target, "val", validation_domain, method, seed)
                    add_task(
                        rows,
                        experiment_id=experiment_id,
                        priority=40,
                        experiment_group="nested",
                        dataset="pacs",
                        target=outer_target,
                        method=method,
                        model="resnet50",
                        seed=seed,
                        selection_protocol="strict_nested_inner",
                        train_domains=",".join(train_domains),
                        validation_domain=validation_domain,
                        save_checkpoint=False,
                    )
    if len(rows) != 232:
        raise RuntimeError(f"expected 232 tasks, built {len(rows)}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_or_create_manifest(path: Path) -> list[dict]:
    expected = build_manifest()
    if not path.exists():
        write_manifest(path, expected)
        return expected
    with path.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    expected_ids = [row["experiment_id"] for row in expected]
    if [row["experiment_id"] for row in existing] != expected_ids:
        raise RuntimeError("existing manifest does not match the locked 232-task protocol")
    refreshed = []
    runtime_fields = ("status", "gpu", "attempts", "exit_code", "error")
    for old, locked in zip(existing, expected):
        if old["status"] == "complete":
            for field in runtime_fields:
                locked[field] = old[field]
        refreshed.append(locked)
    write_manifest(path, refreshed)
    return refreshed


def command_for(row: dict) -> list[str]:
    run_name = row["method"] if row["experiment_group"] not in {"nested", "efficiency"} else row["experiment_id"]
    output_dir = (ROOT / row["result_path"]).parent
    command = [
        sys.executable,
        str(ROOT / "cgc_experiment.py"),
        "--dataset", row["dataset"],
        "--data-root", str(ROOT / DATA_ROOTS[row["dataset"]]),
        "--target", row["target"],
        "--method", row["method"],
        "--run-name", run_name,
        "--model", row["model"],
        "--seed", row["seed"],
        "--epochs", row["epochs"],
        "--batch-size", row["batch_size"],
        "--workers", "4",
        "--image-size", "224",
        "--val-fraction", "0.15",
        "--lr", "0.0003",
        "--weight-decay", "0.0001",
        "--lambda-feat", "0.10",
        "--lambda-kl", "0.05",
        "--temperature", "2",
        "--warmup-epochs", "5",
        "--augmentation-m", "9",
        "--ema-alpha", "0.999",
        "--preprocessing", "timm_vit_standard" if row["model"] == "vit-small" else "legacy_imagenet",
        "--selection-protocol", row["selection_protocol"],
        "--device", "cuda:0",
        "--output", str(output_dir),
        "--diagnostics-output", str(ROOT / row["diagnostics_path"]),
        "--exclusions", str(ROOT / "revision/cross_dataset_exclusions.json"),
    ]
    if row["selection_protocol"] == "strict_nested_inner":
        command.extend([
            "--train-domains", row["train_domains"],
            "--validation-domain", row["validation_domain"],
            "--skip-target-eval",
        ])
    if row["save_checkpoint"] == "1":
        command.append("--save-checkpoint")
    return command


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(row: dict, result: dict) -> None:
    expected_target_evaluations = 0 if row["selection_protocol"] == "strict_nested_inner" else 1
    expected = {
        "dataset": row["dataset"],
        "target": row["target"],
        "method": row["method"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "selection_protocol": row["selection_protocol"],
        "target_evaluations": expected_target_evaluations,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"{key}: expected {value!r}, found {result.get(key)!r}")
    config = result.get("config", {})
    locked = {
        "lambda_feat": 0.10,
        "lambda_kl": 0.05,
        "temperature": 2.0,
        "warmup_epochs": 5,
        "augmentation_N": 2,
        "augmentation_M": 9,
        "gradient_gate_threshold": 0.0,
    }
    for key, value in locked.items():
        if not math.isclose(float(config.get(key, float("nan"))), value, abs_tol=1e-12):
            raise RuntimeError(f"locked config mismatch for {key}")
    if config.get("component_masks_detached") is not True:
        raise RuntimeError("component masks are not recorded as detached")
    if expected_target_evaluations:
        accuracy = result.get("target_accuracy")
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise RuntimeError("invalid target accuracy")
        if row["target"] in result.get("source_domains", []):
            raise RuntimeError("outer target entered source training")
    else:
        if result.get("target_accuracy") is not None or result.get("n_test") != 0:
            raise RuntimeError("nested screening touched outer target")
        if result.get("inner_validation_domain") != row["validation_domain"]:
            raise RuntimeError("inner validation mismatch")
        if set(result.get("source_domains", [])) != set(row["train_domains"].split(",")):
            raise RuntimeError("inner training-domain mismatch")
        if row["target"] in result.get("source_domains", []) or row["target"] == row["validation_domain"]:
            raise RuntimeError("outer target leakage in nested screening")
    if row["method"].startswith("cgc"):
        history = result.get("history", [])
        if len(history) != int(row["epochs"]):
            raise RuntimeError("incomplete diagnostic history")
        for epoch in history:
            for key in ("feature_keep_rate", "kl_keep_rate", "feature_conflict_rate", "kl_conflict_rate"):
                value = float(epoch[key])
                if not 0 <= value <= 1:
                    raise RuntimeError(f"invalid {key}")


def enrich(row: dict, result: dict, command: list[str], gpu: int, started: str, ended: str) -> None:
    result["audit"] = {
        "experiment_id": row["experiment_id"],
        "protocol_version": "cgc-v2-zero-threshold-2026-09-04",
        "target_isolation_passed": True,
        "selection_protocol": row["selection_protocol"],
        "train_domains": result["source_domains"],
        "inner_validation_domain": result.get("inner_validation_domain"),
        "target_evaluations": result["target_evaluations"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "best_source_val_accuracy": result["best_source_val"],
        "command": command,
        "code_sha256": file_sha256(ROOT / "cgc_experiment.py"),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"
        ], text=True).strip(),
        "start_time": started,
        "end_time": ended,
        "wall_clock_minutes": result["elapsed_seconds"] / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revision/cgc_v2_manifest.csv")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    gpu_indices = [int(value) for value in args.gpus.split(",") if value.strip()]
    rows = load_or_create_manifest(ROOT / args.manifest)
    pending: queue.PriorityQueue[tuple[int, int, dict]] = queue.PriorityQueue()
    for index, row in enumerate(rows):
        result_path = ROOT / row["result_path"]
        if row["status"] == "complete" and result_path.is_file():
            validate(row, json.loads(result_path.read_text(encoding="utf-8")))
            continue
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                validate(row, result)
                row["status"], row["exit_code"], row["error"] = "complete", "0", ""
                continue
            except Exception:
                pass
        pending.put((int(row["priority"]), index, row))
    write_manifest(ROOT / args.manifest, rows)
    print(json.dumps({
        "total": len(rows),
        "pending": pending.qsize(),
        "gpus": gpu_indices,
        "workers_per_gpu": args.workers_per_gpu,
        "group_counts": {
            group: sum(row["experiment_group"] == group for row in rows)
            for group in sorted({row["experiment_group"] for row in rows})
        },
    }, indent=2), flush=True)
    if args.dry_run:
        return

    manifest_path = ROOT / args.manifest
    logs = ROOT / "logs/cgc_v2"
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def update(row: dict, **values) -> None:
        with lock:
            row.update({key: str(value) for key, value in values.items()})
            write_manifest(manifest_path, rows)

    def worker(gpu: int, worker_index: int) -> None:
        while True:
            try:
                priority, index, row = pending.get_nowait()
            except queue.Empty:
                return
            command = command_for(row)
            result_path = ROOT / row["result_path"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            attempt = int(row.get("attempts") or 0) + 1
            update(row, status="running", gpu=gpu, attempts=attempt, exit_code="", error="")
            started = now_iso()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            log_path = logs / f"{row['experiment_id']}.log"
            print(f"[GPU {gpu}/{worker_index}] START {row['experiment_id']} attempt={attempt}", flush=True)
            completed = None
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{started}] COMMAND: {' '.join(command)}\n")
                    handle.flush()
                    completed = subprocess.run(
                        command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"trainer exit code {completed.returncode}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                validate(row, result)
                enrich(row, result, command, gpu, started, now_iso())
                temporary = result_path.with_suffix(".json.audit.tmp")
                temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(result_path)
                update(row, status="complete", exit_code=0, error="")
                print(f"[GPU {gpu}/{worker_index}] COMPLETE {row['experiment_id']}", flush=True)
            except Exception as error:
                exit_code = -1 if completed is None else completed.returncode
                if attempt < 2:
                    update(row, status="retry", exit_code=exit_code, error=str(error))
                    pending.put((priority, index, row))
                    print(f"[GPU {gpu}/{worker_index}] RETRY {row['experiment_id']}: {error}", flush=True)
                else:
                    update(row, status="failed", exit_code=exit_code, error=str(error))
                    with lock:
                        failures.append((row["experiment_id"], str(error)))
                    print(f"[GPU {gpu}/{worker_index}] FAILED {row['experiment_id']}: {error}", flush=True)
            finally:
                pending.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu, worker_index), daemon=False)
        for gpu in gpu_indices
        for worker_index in range(args.workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed after retry: {failures[:8]}")
    completed = subprocess.run([sys.executable, str(ROOT / "summarize_cgc_results.py")], cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError("CGC summarization failed")
    print("CGC COMPLETE 232-TASK QUEUE FINISHED", flush=True)


if __name__ == "__main__":
    main()
