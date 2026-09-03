#!/usr/bin/env python3
"""Target-blind SWAD formal matrix for the revision.

This runner uses the official LossValley rule through the local adapter,
source-validation losses only, and exactly one target evaluation.
It has a separate manifest so it cannot collide with the existing matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 3407, 2026, 2027)
DATASETS = {
    "pacs": ("data/PACS", ("Photo", "Art_Painting", "Cartoon", "Sketch")),
    "vlcs": ("data/VLCS", ("Caltech101", "LabelMe", "SUN09", "VOC2007")),
    "officehome": ("data/OfficeHome", ("Art", "Clipart", "Product", "Real_World")),
}
FIELDS = ["experiment_id", "protocol_version", "dataset", "data_root", "outer_target", "method", "seed", "lambda_f", "lambda_k", "temperature", "augmentation_M", "selection_protocol", "config_hash", "status", "result_path", "checkpoint_path", "gpu", "exit_code"]


def sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def create_manifest(path: Path) -> list[dict]:
    rows = []
    for dataset, (data_root, targets) in DATASETS.items():
        for target in targets:
            for seed in SEEDS:
                exp = f"swad-{dataset}-{target}-seed{seed}"
                out = ROOT / "runs" / "revision" / "swad" / dataset / target / str(seed)
                stem = f"{dataset}_{target}_resnet50_swad_seed{seed}"
                cfg = {"dataset": dataset, "target": target, "method": "swad", "seed": seed, "n_converge": 3, "n_tolerance": 6, "tolerance_ratio": 0.3, "averaging": "epoch_segments", "val_fraction": 0.15, "epochs": 30, "batch_size": 64, "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "selection": "source_validation_loss_valley"}
                rows.append({"experiment_id": exp, "protocol_version": "shiftguard-swad-v1.0", "dataset": dataset, "data_root": data_root, "outer_target": target, "method": "swad", "seed": str(seed), "lambda_f": "0.0", "lambda_k": "0.0", "temperature": "2.0", "augmentation_M": "9", "selection_protocol": "source_validation_loss_valley", "config_hash": sha(cfg), "status": "pending", "result_path": str((out / f"{stem}.json").relative_to(ROOT)), "checkpoint_path": str((out / f"{stem}.pt").relative_to(ROOT)), "gpu": "", "exit_code": ""})
    if path.exists():
        with path.open(newline="", encoding="utf-8") as h:
            old = list(csv.DictReader(h))
        if [r["experiment_id"] for r in old] != [r["experiment_id"] for r in rows]:
            raise RuntimeError("existing SWAD manifest IDs differ; refusing overwrite")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, rows)
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def command(row: dict) -> list[str]:
    return [sys.executable, str(ROOT / "shiftguard_multidataset.py"), "--dataset", row["dataset"], "--data-root", str(ROOT / row["data_root"]), "--target", row["outer_target"], "--method", "swad", "--run-name", "swad", "--seed", row["seed"], "--epochs", "30", "--batch-size", "64", "--workers", "8", "--image-size", "224", "--val-fraction", "0.15", "--lr", "0.0003", "--weight-decay", "0.0001", "--lambda-feat", "0.0", "--lambda-kl", "0.0", "--temperature", "2.0", "--warmup-epochs", "5", "--device", "cuda:0", "--output", str((ROOT / row["result_path"]).parent), "--exclusions", str(ROOT / "revision" / "cross_dataset_exclusions.json"), "--save-checkpoint"]


def validate(row: dict, result: dict) -> None:
    for key, expected in {"dataset": row["dataset"], "target": row["outer_target"], "method": "swad", "run_name": "swad", "model": "resnet50", "seed": int(row["seed"]), "target_evaluations": 1}.items():
        if result.get(key) != expected:
            raise RuntimeError(f"{key}: expected {expected!r}, got {result.get(key)!r}")
    if row["outer_target"] in result.get("source_domains", []):
        raise RuntimeError("held-out target appears in source domains")
    acc = result.get("target_accuracy")
    if not isinstance(acc, (int, float)) or not 0 <= acc <= 1:
        raise RuntimeError("invalid target accuracy")


def enrich(row: dict, cmd: list[str], gpu: int, start: str) -> None:
    rp, cp = ROOT / row["result_path"], ROOT / row["checkpoint_path"]
    result = json.loads(rp.read_text(encoding="utf-8"))
    validate(row, result)
    if not cp.is_file():
        raise RuntimeError(f"missing checkpoint {cp}")
    result["audit"] = {"experiment_id": row["experiment_id"], "protocol_version": row["protocol_version"], "config_hash": row["config_hash"], "target_evaluations": 1, "target_accuracy": result["target_accuracy"], "swad_source": "khanrc/swad/domainbed/swad.py:LossValley", "swad_n_converge": 3, "swad_n_tolerance": 6, "swad_tolerance_ratio": 0.3, "averaging_frequency": "one_epoch", "source_only_selection": True, "start_time": start, "end_time": datetime.now(timezone.utc).isoformat()}
    tmp = rp.with_suffix(".audit.tmp"); tmp.write_text(json.dumps(result, indent=2), encoding="utf-8"); tmp.replace(rp)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="revision/swad_manifest.csv"); ap.add_argument("--gpu", nargs=2, type=int, default=(1, 2)); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(); path = ROOT / args.manifest; rows = create_manifest(path)
    pending = [r for r in rows if r["status"] != "complete"]
    print(json.dumps({"total": len(rows), "pending": len(pending), "gpus": args.gpu}, indent=2), flush=True)
    if args.dry_run: return
    qs = queue.Queue(); [qs.put(r) for r in pending]
    logs = ROOT / "logs" / "revision" / "swad"; logs.mkdir(parents=True, exist_ok=True); lock = threading.Lock(); manifest_lock = threading.Lock(); failures = []
    def safe_write():
        with manifest_lock:
            write_manifest(path, rows)
    def worker(gpu: int):
        while True:
            try: row = qs.get_nowait()
            except queue.Empty: return
            i = rows.index(row); rows[i]["status"], rows[i]["gpu"] = "running", str(gpu); safe_write()
            cmd = command(row); start = datetime.now(timezone.utc).isoformat(); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["PYTHONUNBUFFERED"] = "1"; log = logs / f"{row['experiment_id']}.log"
            print(f"[GPU {gpu}] START {row['experiment_id']}", flush=True)
            try:
                with log.open("w", encoding="utf-8") as h:
                    h.write("COMMAND: " + " ".join(cmd) + "\n"); rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=h, stderr=subprocess.STDOUT).returncode
                if rc != 0: raise RuntimeError(f"exit code {rc}")
                enrich(row, cmd, gpu, start); rows[i]["status"], rows[i]["exit_code"] = "complete", "0"; safe_write(); print(f"[GPU {gpu}] COMPLETE {row['experiment_id']}", flush=True)
            except Exception as e:
                rows[i]["status"], rows[i]["exit_code"] = "failed", "-1"; safe_write(); failures.append((row["experiment_id"], str(e))); print(f"[GPU {gpu}] FAILED {row['experiment_id']}: {e}", flush=True)
            finally: qs.task_done()
    ts = [threading.Thread(target=worker, args=(g,)) for g in args.gpu]
    [t.start() for t in ts]; [t.join() for t in ts]
    if failures: raise RuntimeError(str(failures))


if __name__ == "__main__": main()
