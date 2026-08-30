#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:?Usage: ./run_smoke.sh /path/to/PACS}
python shiftguard.py --data-root "$DATA_ROOT" --target Sketch --method erm --seed 42 --epochs 1 --batch-size 16 --no-pretrained
python shiftguard.py --data-root "$DATA_ROOT" --target Sketch --method shiftguard --seed 42 --epochs 1 --batch-size 16 --no-pretrained
python summarize_results.py
