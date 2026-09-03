#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:?Usage: ./run_main_matrix.sh /path/to/PACS}
EPOCHS=${EPOCHS:-30}
for target in Photo Art_Painting Cartoon Sketch; do
  for method in erm aug shiftguard; do
    for seed in 42 123 3407; do
      python shiftguard.py --data-root "$DATA_ROOT" --target "$target" --method "$method" --seed "$seed" --epochs "$EPOCHS"
    done
  done
done
python summarize_results.py
