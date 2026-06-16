#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

# Barlow Twins B-profile (balanced) defaults on GPU 2.
# Override when needed:
# EPOCHS=400 BATCH_SIZE=128 GRAD_ACCUM_STEPS=8 ./scripts/run_barlowtwins_b_gpu2_nohup.sh
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-./outputs/barlowtwins_stl10_b_gpu2_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/barlowtwins_b_gpu2_seed${SEED}.log}"
TB_LOG_DIR="${TB_LOG_DIR:-${SAVE_DIR}/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

echo "Launching Barlow Twins B-profile training with nohup on GPU 2..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python -u pretrain_barlowtwins_stl10.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id 2 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size 96 \
  --projection-hidden-dim 4096 \
  --projection-out-dim 4096 \
  --lambda-offdiag 0.005 \
  --blur-p-view1 1.0 \
  --blur-p-view2 0.1 \
  --solarization-p-view1 0.0 \
  --solarization-p-view2 0.2 \
  --optimizer lars \
  --lr 0.6 \
  --final-lr 0.0006 \
  --momentum 0.9 \
  --weight-decay 0.000001 \
  --warmup-epochs 10 \
  --amp \
  --tensorboard \
  --eval-interval 50 \
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
