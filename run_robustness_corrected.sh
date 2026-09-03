#!/usr/bin/env bash
set -euo pipefail
mkdir -p runs/corrected_robustness logs/corrected_robustness
seeds=(42 123 3407)
for gpu in 0 1 2; do
  seed=${seeds[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu python3 robustness_corrected.py --seed "$seed" --device cuda:0 --output "runs/corrected_robustness/seed${seed}.csv" >"logs/corrected_robustness/seed${seed}.log" 2>&1 &
done
wait
python3 - <<'INNER'
import csv
from pathlib import Path
rows=[]
for p in sorted(Path('runs/corrected_robustness').glob('seed*.csv')): rows.extend(csv.DictReader(p.open()))
with open('runs/corrected_robustness/results.csv','w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['seed','method','corruption','severity','accuracy']); w.writeheader(); w.writerows(rows)
print('rows',len(rows))
INNER
