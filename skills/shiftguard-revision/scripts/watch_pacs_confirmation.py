#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from revision_common import project_root


def main() -> None:
    root = project_root()
    manifest = root / "revision" / "revision_manifest.csv"
    log = root / "revision" / "pacs5_watcher_status.json"
    while True:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts = {status: sum(row["status"] == status for row in rows) for status in ("pending", "running", "complete", "failed")}
        payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "counts": counts}
        temporary = log.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(log)
        print(json.dumps(payload), flush=True)
        if counts["failed"]:
            raise RuntimeError("one or more formal jobs failed; summary not generated")
        if counts["complete"] == len(rows):
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("summarize_revision.py"))],
                cwd=root,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"summary exited with {completed.returncode}")
            print("AUDITED FIVE-SEED SUMMARY COMPLETE", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
