#!/usr/bin/env bash
# SimCLR exp4 settings (train_config.json) with epochs=500 on GPU 0.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-ab5g_simclr_no_lars_exp4_e500_seed${SEED}_e500}"
SAVE_DIR="${SAVE_DIR:-./outputs/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-./logs/${RUN_NAME}.log}"

# Matches outputs/ab5g_simclr_no_lars_exp4_e400_seed42_e400/train_config.json (+ epochs 500, gpu 0)
echo "[launch] gpu=${GPU_ID} run=${RUN_NAME}"
echo "  save_dir=${SAVE_DIR}"
echo "  log=${LOG_FILE}"

nohup python -u pretrain_simclr_stl10.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id "${GPU_ID}" \
  --epochs 500 \
  --batch-size 320 \
  --num-workers 8 \
  --image-size 96 \
  --temperature 0.2 \
  --projection-hidden-dim 2048 \
  --projection-out-dim 256 \
  --lr 0.3 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --warmup-epochs 10 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --eval-batch-size 256 \
  --linear-eval-epochs 100 \
  --linear-eval-lr 0.01 \
  --knn-k 20 \
  --knn-temperature 0.1 \
  --periodic-keep-last 3 \
  --log-interval 20 \
  --save-every 25 \
  --disable-tqdm \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"
echo "  pid=${PID}"
echo "Monitor: tail -f ${LOG_FILE}"
