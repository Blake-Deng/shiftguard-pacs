#!/usr/bin/env bash
set -u

WORKER=${1:?worker id 0, 1, or 2}
mkdir -p runs logs
idx=0
for target in Photo Art_Painting Cartoon Sketch; do
  for method in erm aug shiftguard; do
    for seed in 42 123 3407; do
      slot=$((idx % 3))
      if [ "$slot" = "$WORKER" ]; then
        stamp=$(date +%Y%m%d_%H%M%S)
        echo "[$stamp] worker=$WORKER gpu=$WORKER target=$target method=$method seed=$seed" | tee -a "logs/worker${WORKER}.log"
        CUDA_VISIBLE_DEVICES="$WORKER" python3 shiftguard.py \
          --data-root data/PACS --target "$target" --method "$method" --seed "$seed" \
          --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-64}" --workers "${WORKERS:-8}" \
          --device cuda:0 --output runs >>"logs/worker${WORKER}.log" 2>&1
        code=$?
        echo "[$(date +%Y%m%d_%H%M%S)] finished code=$code target=$target method=$method seed=$seed" | tee -a "logs/worker${WORKER}.log"
      fi
      idx=$((idx + 1))
    done
  done
done
echo "worker $WORKER complete" | tee -a "logs/worker${WORKER}.log"
