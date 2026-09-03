#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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

from create_manifest import FIELDS
from revision_common import DOMAINS, file_sha256, project_root
from validate_result import validate


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 P0 jobs, found {len(rows)}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_for(root: Path, row: dict) -> list[str]:
    output_dir = (root / row["result_path"]).parent
    return [
        sys.executable, str(root / "shiftguard_corrected.py"),
        "--data-root", str(root / "data" / "PACS"),
        "--target", row["outer_target"],
        "--method", row["trainer_method"],
        "--run-name", row["method"],
        "--model", row["backbone"],
        "--seed", row["seed"],
        "--epochs", "30",
        "--batch-size", "64",
        "--workers", "8",
        "--image-size", "224",
        "--val-fraction", "0.15",
        "--lr", "0.0003",
        "--weight-decay", "0.0001",
        "--lambda-feat", row["lambda_f"],
        "--lambda-kl", row["lambda_k"],
        "--temperature", row["temperature"],
        "--gate-tau", "0.5",
        "--weight-min", "0.5",
        "--weight-max", "2.0",
        "--warmup-epochs", "5",
        "--device", "cuda:0",
        "--output", str(output_dir),
        "--save-checkpoint",
    ]


def best_epoch(result: dict) -> int:
    history = result.get("history", [])
    if not history:
        raise RuntimeError("result has no training history")
    best = max(float(item["val_accuracy"]) for item in history)
    return next(int(item["epoch"]) for item in history if float(item["val_accuracy"]) == best)


def code_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable:not-a-git-worktree"


def gpu_name(index: int) -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()


def enrich_result(root: Path, row: dict, command: list[str], gpu: int, started: str, ended: str) -> None:
    result_path = root / row["result_path"]
    checkpoint_path = root / row["checkpoint_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate(row, result)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"missing checkpoint: {checkpoint_path}")
    elapsed = float(result["elapsed_seconds"])
    result["audit"] = {
        "experiment_id": row["experiment_id"],
        "protocol_version": row["protocol_version"],
        "config_hash": row["config_hash"],
        "dataset_fingerprint": row["dataset_fingerprint"],
        "train_domains": [domain for domain in DOMAINS if domain != row["outer_target"]],
        "optimization_seed": int(row["optimization_seed"]),
        "source_split_seed": int(row["source_split_seed"]),
        "checkpoint_epoch": best_epoch(result),
        "best_source_val_accuracy": float(result["best_source_val"]),
        "target_evaluations": int(result["target_evaluations"]),
        "target_accuracy": float(result["target_accuracy"]),
        "command": command,
        "code_commit": code_commit(root),
        "code_sha256": file_sha256(root / "shiftguard_corrected.py"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": gpu_name(gpu),
        "start_time": started,
        "end_time": ended,
        "wall_clock_minutes": elapsed / 60.0,
        "exit_code": 0,
        "rerun_reason": row.get("rerun_reason", ""),
    }
    temporary = result_path.with_suffix(".json.audit.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    validate(row, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revision/revision_manifest.csv")
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    manifest_path = root / args.manifest
    rows = load_manifest(manifest_path)
    gpu_indices = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpu_indices:
        raise RuntimeError("at least one GPU is required")

    pending: queue.Queue[dict] = queue.Queue()
    for row in rows:
        result_path = root / row["result_path"]
        if row["status"] == "complete":
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate(row, result)
            continue
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("audit", {}).get("config_hash") == row["config_hash"]:
                validate(row, result)
                row["status"] = "complete"
                row["exit_code"] = "0"
                continue
        pending.put(row)
    write_manifest(manifest_path, rows)

    if args.dry_run:
        print(json.dumps({
            "pending": pending.qsize(),
            "gpus": gpu_indices,
            "commands": [command_for(root, row) for row in list(pending.queue)[:2]],
        }, indent=2))
        return

    logs = root / "logs" / "revision" / "pacs5"
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures = []

    def update(row: dict, status: str, gpu: int, exit_code: int | str) -> None:
        with lock:
            row["status"] = status
            row["gpu"] = str(gpu)
            row["exit_code"] = str(exit_code)
            write_manifest(manifest_path, rows)

    def worker(gpu: int) -> None:
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return
            experiment_id = row["experiment_id"]
            command = command_for(root, row)
            result_path = root / row["result_path"]
            result_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"{experiment_id}.log"
            started = now_iso()
            update(row, "running", gpu, "")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            print(f"[GPU {gpu}] START {experiment_id}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as handle:
                    handle.write("COMMAND: " + " ".join(command) + "\n")
                    handle.flush()
                    completed = subprocess.run(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
                ended = now_iso()
                if completed.returncode != 0:
                    raise RuntimeError(f"trainer exit code {completed.returncode}")
                enrich_result(root, row, command, gpu, started, ended)
                update(row, "complete", gpu, 0)
                print(f"[GPU {gpu}] COMPLETE {experiment_id}", flush=True)
            except Exception as error:
                code = completed.returncode if "completed" in locals() else -1
                update(row, "failed", gpu, code)
                with lock:
                    failures.append((experiment_id, str(error), str(log_path)))
                print(f"[GPU {gpu}] FAILED {experiment_id}: {error}", flush=True)
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_indices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")
    print("PACS FIVE-SEED EXTENSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
