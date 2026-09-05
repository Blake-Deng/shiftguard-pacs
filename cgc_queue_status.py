#!/usr/bin/env python3
"""Print concise progress for the restartable CGC v2 queue."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "revision/cgc_v2_manifest.csv"


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    statuses = Counter(row["status"] for row in rows)
    groups = defaultdict(Counter)
    durations = []
    for row in rows:
        groups[row["experiment_group"]][row["status"]] += 1
        if row["status"] == "complete":
            path = ROOT / row["result_path"]
            if path.is_file():
                durations.append(json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"] / 60.0)
    print(f"TOTAL {len(rows)} | " + " | ".join(f"{key} {value}" for key, value in sorted(statuses.items())))
    for group in sorted(groups):
        counts = groups[group]
        print(f"{group:12s} " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    running = [row for row in rows if row["status"] == "running"]
    if running:
        print("\nRUNNING")
        for row in running:
            print(f"GPU {row['gpu']}  {row['experiment_id']}  attempt={row['attempts']}")
    if durations:
        print(f"\nCompleted duration: median={statistics.median(durations):.1f} min, mean={statistics.fmean(durations):.1f} min")
    if statuses.get("failed"):
        print("\nFAILED")
        for row in rows:
            if row["status"] == "failed":
                print(f"{row['experiment_id']}: {row['error']}")


if __name__ == "__main__":
    main()
