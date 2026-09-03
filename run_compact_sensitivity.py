#!/usr/bin/env python3
"""Run the predeclared PACS single-seed OAT sensitivity matrix."""
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "skills" / "shiftguard-revision" / "scripts"))
from revision_common import DOMAINS, canonical_hash, file_sha256, pacs_inventory  # noqa: E402

PROTOCOL = "shiftguard-compact-sensitivity-v1.0"
SEED = 42
FIELDS = [
    "experiment_id", "experiment_group", "protocol_version", "dataset", "data_root",
    "outer_target", "method", "trainer_method", "backbone", "seed", "optimization_seed",
    "source_split_seed", "sweep", "factor_value", "lambda_f", "lambda_k", "temperature",
    "augmentation_M", "selection_protocol", "config_hash", "dataset_fingerprint", "status",
    "result_path", "checkpoint_path", "gpu", "exit_code", "rerun_reason",
]
SETTINGS = (
    ("lambda_f", "0.05", 0.05, 0.05, 2.0, 9, "lambda_f_0p05"),
    ("lambda_f", "0.20", 0.20, 0.05, 2.0, 9, "lambda_f_0p20"),
    ("lambda_k", "0.025", 0.10, 0.025, 2.0, 9, "lambda_k_0p025"),
    ("lambda_k", "0.10", 0.10, 0.10, 2.0, 9, "lambda_k_0p10"),
    ("temperature", "1", 0.10, 0.05, 1.0, 9, "temperature_1"),
    ("temperature", "4", 0.10, 0.05, 4.0, 9, "temperature_4"),
    ("augmentation_M", "5", 0.10, 0.05, 2.0, 5, "augmentation_M_5"),
    ("augmentation_M", "13", 0.10, 0.05, 2.0, 13, "augmentation_M_13"),
)


def config(target: str, setting: tuple, fingerprint: str) -> dict:
    sweep, factor_value, lf, lk, temperature, magnitude, run_name = setting
    return {
        "protocol_version": PROTOCOL, "dataset": "PACS", "outer_target": target,
        "train_domains": [d for d in DOMAINS if d != target], "method": "feature_plus_kl",
        "trainer_method": "feat_kl", "run_name": run_name, "backbone": "resnet50",
        "pretrained": True, "seed": SEED, "optimization_seed": SEED, "source_split_seed": SEED,
        "source_val_fraction": 0.15, "epochs": 30, "batch_size": 64, "image_size": 224,
        "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "scheduler": "cosine",
        "lambda_f": lf, "lambda_k": lk, "temperature": temperature,
        "consistency_ramp_epochs": 5, "augmentation_N": 2, "augmentation_M": magnitude,
        "sweep": sweep, "factor_value": factor_value,
        "checkpoint_rule": "highest_source_validation_accuracy_earliest_exact_tie",
        "target_evaluations": 1, "dataset_fingerprint": fingerprint,
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def create_or_load_manifest(path: Path) -> list[dict]:
    fingerprint, _ = pacs_inventory(ROOT / "data" / "PACS")
    rows = []
    for setting in SETTINGS:
        for target in DOMAINS:
            cfg = config(target, setting, fingerprint)
            run_name = setting[-1]
            experiment_id = f"sensitivity-{run_name}-{target}-seed{SEED}"
            out = ROOT / "runs" / "revision" / "sensitivity" / setting[0] / run_name / target / str(SEED)
            stem = f"{target}_resnet50_{run_name}_seed{SEED}"
            rows.append({
                "experiment_id": experiment_id, "experiment_group": "P1_compact_sensitivity",
                "protocol_version": PROTOCOL, "dataset": "PACS", "data_root": "data/PACS",
                "outer_target": target, "method": "feature_plus_kl", "trainer_method": "feat_kl",
                "backbone": "torchvision:resnet50:IMAGENET1K_V2", "seed": str(SEED),
                "optimization_seed": str(SEED), "source_split_seed": str(SEED),
                "sweep": setting[0], "factor_value": setting[1], "lambda_f": str(setting[2]),
                "lambda_k": str(setting[3]), "temperature": str(setting[4]),
                "augmentation_M": str(setting[5]),
                "selection_protocol": "highest_source_validation_accuracy_earliest_exact_tie",
                "config_hash": canonical_hash(cfg), "dataset_fingerprint": fingerprint,
                "status": "pending", "result_path": str((out / f"{stem}.json").relative_to(ROOT)),
                "checkpoint_path": str((out / f"{stem}.pt").relative_to(ROOT)), "gpu": "",
                "exit_code": "", "rerun_reason": "",
            })
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
        if [r["experiment_id"] for r in old] != [r["experiment_id"] for r in rows]:
            raise RuntimeError("existing sensitivity manifest differs; refusing to overwrite")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, rows)
    return rows


def command_for(row: dict) -> list[str]:
    return [
        sys.executable, str(ROOT / "shiftguard_corrected.py"), "--data-root", str(ROOT / row["data_root"]),
        "--target", row["outer_target"], "--method", "feat_kl", "--run-name",
        row["experiment_id"].split("-", 1)[1].rsplit("-", 2)[0], "--model", "resnet50",
        "--seed", row["seed"], "--epochs", "30", "--batch-size", "64", "--workers", "8",
        "--image-size", "224", "--preprocessing", "legacy_imagenet", "--val-fraction", "0.15",
        "--lr", "0.0003", "--weight-decay", "0.0001", "--lambda-feat", row["lambda_f"],
        "--lambda-kl", row["lambda_k"], "--temperature", row["temperature"],
        "--augmentation-m", row["augmentation_M"], "--warmup-epochs", "5", "--device", "cuda:0",
        "--output", str((ROOT / row["result_path"]).parent), "--save-checkpoint",
    ]


def validate(row: dict, result: dict) -> None:
    expected = {"target": row["outer_target"], "method": "feat_kl", "model": "resnet50", "seed": SEED, "target_evaluations": 1}
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"{key}: expected {value!r}, found {result.get(key)!r}")
    accuracy = result.get("target_accuracy")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise RuntimeError("invalid target accuracy")
    actual = result.get("config", {})
    for key, value in {"lambda_feat": float(row["lambda_f"]), "lambda_kl": float(row["lambda_k"]), "temperature": float(row["temperature"]), "augmentation_M": int(row["augmentation_M"]), "warmup_epochs": 5}.items():
        if abs(float(actual.get(key, float("nan"))) - value) > 1e-12:
            raise RuntimeError(f"config {key}: expected {value}, found {actual.get(key)!r}")


def best_epoch(result: dict) -> int:
    history = result["history"]
    best = max(float(item["val_accuracy"]) for item in history)
    return next(int(item["epoch"]) for item in history if float(item["val_accuracy"]) == best)


def gpu_name(index: int) -> str:
    return subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"], text=True).strip()


def enrich(row: dict, cmd: list[str], gpu: int, started: str) -> None:
    result_path, checkpoint_path = ROOT / row["result_path"], ROOT / row["checkpoint_path"]
    result = json.loads(result_path.read_text(encoding="utf-8")); validate(row, result)
    if not checkpoint_path.is_file(): raise RuntimeError(f"missing checkpoint: {checkpoint_path}")
    result["audit"] = {
        "experiment_id": row["experiment_id"], "protocol_version": PROTOCOL,
        "config_hash": row["config_hash"], "dataset_fingerprint": row["dataset_fingerprint"],
        "train_domains": [d for d in DOMAINS if d != row["outer_target"]],
        "optimization_seed": SEED, "source_split_seed": SEED, "checkpoint_epoch": best_epoch(result),
        "best_source_val_accuracy": float(result["best_source_val"]), "target_evaluations": 1,
        "target_accuracy": float(result["target_accuracy"]), "command": cmd,
        "code_commit": "unavailable:not-a-git-worktree", "code_sha256": file_sha256(ROOT / "shiftguard_corrected.py"),
        "checkpoint_sha256": file_sha256(checkpoint_path), "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "timm_version": timm.__version__,
        "cuda_version": torch.version.cuda, "gpu_model": gpu_name(gpu), "start_time": started,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "wall_clock_minutes": float(result["elapsed_seconds"]) / 60.0, "exit_code": 0,
        "rerun_reason": row.get("rerun_reason", ""),
    }
    temporary = result_path.with_suffix(".json.audit.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8"); temporary.replace(result_path)
    validate(row, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revision/sensitivity_manifest.csv")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1: raise ValueError("workers-per-gpu must be positive")
    manifest_path = ROOT / args.manifest
    rows = create_or_load_manifest(manifest_path)
    work = queue.Queue()
    for row in rows:
        result_path = ROOT / row["result_path"]
        if row["status"] == "complete":
            validate(row, json.loads(result_path.read_text(encoding="utf-8"))); continue
        work.put(row)
    print(json.dumps({"total_new_runs": len(rows), "pending": work.qsize(), "gpu": args.gpu, "workers": args.workers_per_gpu}, indent=2), flush=True)
    if args.dry_run: return
    logs = ROOT / "logs" / "revision" / "sensitivity"; logs.mkdir(parents=True, exist_ok=True)
    manifest_lock, failures = threading.Lock(), []
    def update(row: dict, status: str, exit_code: str = "") -> None:
        with manifest_lock:
            row["status"], row["gpu"], row["exit_code"] = status, str(args.gpu), exit_code
            write_manifest(manifest_path, rows)
    def worker() -> None:
        while True:
            try: row = work.get_nowait()
            except queue.Empty: return
            cmd = command_for(row); started = datetime.now(timezone.utc).isoformat(); update(row, "running")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env["PYTHONUNBUFFERED"] = "1"
            print(f"[GPU {args.gpu}] START {row['experiment_id']}", flush=True)
            try:
                with (logs / f"{row['experiment_id']}.log").open("w", encoding="utf-8") as handle:
                    handle.write("COMMAND: " + " ".join(cmd) + "\n"); handle.flush()
                    completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if completed.returncode != 0: raise RuntimeError(f"trainer exit code {completed.returncode}")
                enrich(row, cmd, args.gpu, started); update(row, "complete", "0")
                print(f"[GPU {args.gpu}] COMPLETE {row['experiment_id']}", flush=True)
            except Exception as error:
                update(row, "failed", "-1"); failures.append((row["experiment_id"], str(error)))
                print(f"[GPU {args.gpu}] FAILED {row['experiment_id']}: {error}", flush=True)
            finally: work.task_done()
    threads = [threading.Thread(target=worker, daemon=False) for _ in range(args.workers_per_gpu)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures: raise RuntimeError(f"failed jobs: {failures}")
    completed = subprocess.run([sys.executable, str(ROOT / "summarize_compact_sensitivity.py")], cwd=ROOT)
    if completed.returncode != 0: raise RuntimeError("sensitivity summary failed")
    print("COMPACT SENSITIVITY COMPLETE", flush=True)


if __name__ == "__main__": main()
