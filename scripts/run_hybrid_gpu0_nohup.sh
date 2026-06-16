#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

# You can override these via environment variables when launching:
#   EPOCHS=200 BATCH_SIZE=256 SAVE_DIR=... ./scripts/run_hybrid_gpu0_nohup.sh
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-./outputs/simclr_stl10_hybrid_gpu0_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/simclr_hybrid_gpu0_seed${SEED}.log}"

# Hybrid defaults (override if needed)
MASK_RATIO="${MASK_RATIO:-0.3}"
MASK_PATCH_SIZE="${MASK_PATCH_SIZE:-16}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
EMA_MOMENTUM="${EMA_MOMENTUM:-0.996}"

echo "Launching hybrid training with nohup on GPU 0..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python pretrain_simclr_stl10_hybrid.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id 0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size 96 \
  --temperature 0.2 \
  --projection-hidden-dim 2048 \
  --projection-out-dim 128 \
  --lr 0.3 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --warmup-epochs 10 \
  --amp \
  --tensorboard \
  --eval-interval 25 \
  --eval-batch-size 256 \
  --linear-eval-epochs 10 \
  --linear-eval-lr 0.1 \
  --knn-k 20 \
  --knn-temperature 0.1 \
  --periodic-keep-last 3 \
  --log-interval 20 \
  --save-every 25 \
  --disable-tqdm \
  --mask-ratio "${MASK_RATIO}" \
  --mask-patch-size "${MASK_PATCH_SIZE}" \
  --distill-weight "${DISTILL_WEIGHT}" \
  --ema-momentum "${EMA_MOMENTUM}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"

echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
