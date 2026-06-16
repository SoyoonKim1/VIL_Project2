#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

# Barlow Twins B-profile defaults on GPU 7.
# Override example:
# SEED=42 EPOCHS=600 BATCH_SIZE=128 GRAD_ACCUM_STEPS=8 LR=0.6 \
#   bash scripts/run_barlowtwins_b_gpu7_efflr06_nohup.sh
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-7}"
LR="${LR:-0.6}"
FINAL_LR="${FINAL_LR:-0.0006}"
SCALE_LR_BY_BATCH="${SCALE_LR_BY_BATCH:-false}"

SAVE_DIR="${SAVE_DIR:-./outputs/barlowtwins_stl10_b_gpu${GPU_ID}_efflr06_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/barlowtwins_b_gpu${GPU_ID}_efflr06_seed${SEED}.log}"
TB_LOG_DIR="${TB_LOG_DIR:-${SAVE_DIR}/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

SCALE_FLAG=""
if [[ "${SCALE_LR_BY_BATCH}" == "true" ]]; then
  SCALE_FLAG="--scale-lr-by-batch"
fi

echo "Launching Barlow Twins B-profile training with nohup on GPU ${GPU_ID}..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python -u pretrain_barlowtwins_stl10.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id "${GPU_ID}" \
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
  --lr "${LR}" \
  --final-lr "${FINAL_LR}" \
  ${SCALE_FLAG} \
  --momentum 0.9 \
  --weight-decay 0.000001 \
  --warmup-epochs 10 \
  --amp \
  --tensorboard \
  --eval-interval 50 \
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

echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT} --host 0.0.0.0"
