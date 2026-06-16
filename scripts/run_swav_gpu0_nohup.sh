#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

# You can override these via environment variables:
#   EPOCHS=200 BATCH_SIZE=256 SAVE_DIR=... ./scripts/run_swav_gpu0_nohup.sh
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-./outputs/swav_stl10_gpu0_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/swav_gpu0_seed${SEED}.log}"
TB_LOG_DIR="${TB_LOG_DIR:-${SAVE_DIR}/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

# SwAV defaults (override if needed)
NUM_PROTOTYPES="${NUM_PROTOTYPES:-3000}"
TEMPERATURE="${TEMPERATURE:-0.1}"
SINKHORN_EPSILON="${SINKHORN_EPSILON:-0.05}"
SINKHORN_ITERATIONS="${SINKHORN_ITERATIONS:-3}"
# Space-separated crop ids, e.g. "0 1"
ASSIGNMENT_CROP_IDS="${ASSIGNMENT_CROP_IDS:-0 1}"

echo "Launching SwAV training with nohup on GPU 0..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python pretrain_swav_stl10.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id 0 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size 96 \
  --projection-hidden-dim 2048 \
  --projection-out-dim 128 \
  --num-prototypes "${NUM_PROTOTYPES}" \
  --temperature "${TEMPERATURE}" \
  --sinkhorn-epsilon "${SINKHORN_EPSILON}" \
  --sinkhorn-iterations "${SINKHORN_ITERATIONS}" \
  --assignment-crop-ids ${ASSIGNMENT_CROP_IDS} \
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
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"

echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT} --host 0.0.0.0"
