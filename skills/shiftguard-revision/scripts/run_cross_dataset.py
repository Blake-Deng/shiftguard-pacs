#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from create_cross_manifest import DATASETS, FIELDS
from revision_common import file_sha256, project_root


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 cross-dataset jobs, found {len(rows)}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_for(root: Path, row: dict) -> list[str]:
    return [
        sys.executable, str(root / "shiftguard_multidataset.py"),
        "--dataset", row["dataset"], "--data-root", str(root / row["data_root"]),
        "--target", row["outer_target"], "--method", row["trainer_method"],
        "--run-name", row["method"], "--seed", row["seed"],
        "--epochs", "30", "--batch-size", "64", "--workers", "8",
        "--image-size", "224", "--val-fraction", "0.15",
        "--lr", "0.0003", "--weight-decay", "0.0001",
        "--lambda-feat", row["lambda_f"], "--lambda-kl", row["lambda_k"],
        "--temperature", row["temperature"], "--warmup-epochs", "5",
        "--device", "cuda:0", "--output", str((root / row["result_path"]).parent),
        "--exclusions", str(root / "revision" / "cross_dataset_exclusions.json"),
        "--save-checkpoint",
    ]


def validate(row: dict, result: dict) -> None:
    expected = {
        "dataset": row["dataset"], "target": row["outer_target"],
        "method": row["trainer_method"], "run_name": row["method"],
        "model": "resnet50", "seed": int(row["seed"]), "target_evaluations": 1,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"{key}: expected {value!r}, found {result.get(key)!r}")
    accuracy = result.get("target_accuracy")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise RuntimeError(f"invalid target accuracy: {accuracy!r}")
    expected_sources = [target for target in DATASETS[row["dataset"]]["targets"] if target != row["outer_target"]]
    if result.get("source_domains") != expected_sources or row["outer_target"] in result.get("source_domains", []):
        raise RuntimeError("source-domain audit failed")
    config = result.get("config", {})
    numeric = {
        "val_fraction": 0.15, "lambda_feat": float(row["lambda_f"]),
        "lambda_kl": float(row["lambda_k"]), "temperature": 2.0,
        "warmup_epochs": 5, "augmentation_M": 9,
    }
    for key, value in numeric.items():
        if abs(float(config.get(key, float("nan"))) - value) > 1e-12:
            raise RuntimeError(f"config {key}: expected {value}, found {config.get(key)!r}")
    if config.get("preprocessing") != "legacy_imagenet":
        raise RuntimeError("preprocessing mismatch")
    exclusions = json.loads((project_root() / "revision" / "cross_dataset_exclusions.json").read_text(encoding="utf-8"))
    expected_excluded = len(exclusions["datasets"][row["dataset"]])
    if config.get("exclusion_policy") != exclusions["policy_version"]:
        raise RuntimeError("exclusion policy mismatch")
    if int(config.get("excluded_images", -1)) != expected_excluded:
        raise RuntimeError("excluded image count mismatch")
    audit = result.get("audit")
    if audit is not None and audit.get("config_hash") != row["config_hash"]:
        raise RuntimeError("audit config hash differs from manifest")


def best_epoch(result: dict) -> int:
    history = result.get("history", [])
    best = max(float(item["val_accuracy"]) for item in history)
    return next(int(item["epoch"]) for item in history if float(item["val_accuracy"]) == best)


def code_commit(root: Path) -> str:
    completed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable:not-a-git-worktree"


def gpu_name(index: int) -> str:
    return subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"], text=True).strip()


def enrich(root: Path, row: dict, command: list[str], gpu: int, started: str, ended: str) -> None:
    result_path, checkpoint_path = root / row["result_path"], root / row["checkpoint_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate(row, result)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"missing checkpoint: {checkpoint_path}")
    result["audit"] = {
        "experiment_id": row["experiment_id"], "protocol_version": row["protocol_version"],
        "config_hash": row["config_hash"], "dataset_fingerprint": row["dataset_fingerprint"],
        "train_domains": result["source_domains"],
        "optimization_seed": int(row["optimization_seed"]), "source_split_seed": int(row["source_split_seed"]),
        "checkpoint_epoch": best_epoch(result), "best_source_val_accuracy": float(result["best_source_val"]),
        "target_evaluations": 1, "target_accuracy": float(result["target_accuracy"]),
        "command": command, "code_commit": code_commit(root),
        "code_sha256": file_sha256(root / "shiftguard_multidataset.py"),
        "shared_core_sha256": file_sha256(root / "shiftguard_corrected.py"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pytorch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__, "cuda_version": torch.version.cuda,
        "gpu_model": gpu_name(gpu), "start_time": started, "end_time": ended,
        "wall_clock_minutes": float(result["elapsed_seconds"]) / 60.0,
        "exit_code": 0, "rerun_reason": row.get("rerun_reason", ""),
    }
    temporary = result_path.with_suffix(".json.audit.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    validate(row, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revision/cross_dataset_manifest.csv")
    parser.add_argument("--vlcs-gpu", type=int, default=1)
    parser.add_argument("--officehome-gpu", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be at least 1")
    root = project_root()
    manifest_path = root / args.manifest
    rows = load_manifest(manifest_path)
    queues = {"vlcs": queue.Queue(), "officehome": queue.Queue()}
    for row in rows:
        result_path = root / row["result_path"]
        if row["status"] == "complete":
            validate(row, json.loads(result_path.read_text(encoding="utf-8")))
            continue
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("audit", {}).get("config_hash") == row["config_hash"]:
                validate(row, result)
                row["status"], row["exit_code"] = "complete", "0"
                continue
        queues[row["dataset"]].put(row)
    write_manifest(manifest_path, rows)
    if args.dry_run:
        print(json.dumps({
            "pending": {dataset: work.qsize() for dataset, work in queues.items()},
            "gpus": {"vlcs": args.vlcs_gpu, "officehome": args.officehome_gpu},
            "workers_per_gpu": args.workers_per_gpu,
            "sample_commands": {dataset: command_for(root, list(work.queue)[0]) for dataset, work in queues.items() if work.qsize()},
        }, indent=2))
        return
    logs = root / "logs" / "revision" / "cross_dataset"
    logs.mkdir(parents=True, exist_ok=True)
    lock, failures = threading.Lock(), []

    def update(row: dict, status: str, gpu: int, exit_code: int | str) -> None:
        with lock:
            row["status"], row["gpu"], row["exit_code"] = status, str(gpu), str(exit_code)
            write_manifest(manifest_path, rows)

    def get_next(primary: str):
        secondary = "officehome" if primary == "vlcs" else "vlcs"
        for dataset in (primary, secondary):
            try:
                return queues[dataset], queues[dataset].get_nowait()
            except queue.Empty:
                pass
        return None, None

    def worker(gpu: int, primary: str) -> None:
        while True:
            source_queue, row = get_next(primary)
            if row is None:
                return
            experiment_id, command = row["experiment_id"], command_for(root, row)
            (root / row["result_path"]).parent.mkdir(parents=True, exist_ok=True)
            started, completed = now_iso(), None
            update(row, "running", gpu, "")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"], env["PYTHONUNBUFFERED"] = str(gpu), "1"
            print(f"[GPU {gpu}] START {experiment_id}", flush=True)
            try:
                with (logs / f"{experiment_id}.log").open("w", encoding="utf-8") as handle:
                    handle.write("COMMAND: " + " ".join(command) + "\n")
                    handle.flush()
                    completed = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if completed.returncode != 0:
                    raise RuntimeError(f"trainer exit code {completed.returncode}")
                enrich(root, row, command, gpu, started, now_iso())
                update(row, "complete", gpu, 0)
                print(f"[GPU {gpu}] COMPLETE {experiment_id}", flush=True)
            except Exception as error:
                code = -1 if completed is None else completed.returncode
                update(row, "failed", gpu, code)
                with lock:
                    failures.append((experiment_id, str(error)))
                print(f"[GPU {gpu}] FAILED {experiment_id}: {error}", flush=True)
            finally:
                source_queue.task_done()

    threads = []
    for worker_index in range(args.workers_per_gpu):
        threads.append(threading.Thread(
            target=worker, args=(args.vlcs_gpu, "vlcs"),
            name=f"vlcs-gpu{args.vlcs_gpu}-worker{worker_index}", daemon=False,
        ))
        threads.append(threading.Thread(
            target=worker, args=(args.officehome_gpu, "officehome"),
            name=f"officehome-gpu{args.officehome_gpu}-worker{worker_index}", daemon=False,
        ))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_cross.py"))], cwd=root)
    if completed.returncode != 0:
        raise RuntimeError("cross-dataset summary failed")
    print("CROSS-DATASET FORMAL MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
