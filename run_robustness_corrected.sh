#!/usr/bin/env bash
set -euo pipefail
mkdir -p runs/corrected_robustness logs/corrected_robustness
gpu_spec="${SHIFTGUARD_GPUS:-0}"
IFS=',' read -r -a gpus <<< "$gpu_spec"
seeds=(42 123 3407)

run_seed() {
  local seed=$1
  local gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" python3 robustness_corrected.py \
    --seed "$seed" --device cuda:0 \
    --output "runs/corrected_robustness/seed${seed}.csv" \
    >"logs/corrected_robustness/seed${seed}.log" 2>&1
}

if [ "${#gpus[@]}" -ge 3 ]; then
  for index in 0 1 2; do
    run_seed "${seeds[$index]}" "${gpus[$index]}" &
  done
  wait
else
  for index in 0 1 2; do
    gpu_index=$((index % ${#gpus[@]}))
    run_seed "${seeds[$index]}" "${gpus[$gpu_index]}"
  done
fi

python3 - <<'PY'
import csv
from pathlib import Path
rows = []
for path in sorted(Path("runs/corrected_robustness").glob("seed*.csv")):
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
with Path("runs/corrected_robustness/results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["seed", "method", "corruption", "severity", "accuracy"])
    writer.writeheader()
    writer.writerows(rows)
print("rows", len(rows))
PY
