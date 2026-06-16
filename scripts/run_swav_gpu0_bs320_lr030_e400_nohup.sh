#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

# Same recipe as swav_gpu6_bs512_lr015_e400_proto512, except batch/lr/gpu.
# eff_lr = lr * (batch_size / 256) = 0.30 * (320 / 256) = 0.375 (matches SimCLR exp4 scaling)
#
# Override example:
#   SEED=42 GPU_ID=0 bash scripts/run_swav_gpu0_bs320_lr030_e400_nohup.sh

SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-320}"
LR="${LR:-0.30}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-512}"

RUN_TAG="gpu${GPU_ID}_bs${BATCH_SIZE}_lr030_e${EPOCHS}_proto${NUM_PROTOTYPES}"
SAVE_DIR="${SAVE_DIR:-./outputs/swav_stl10_${RUN_TAG}_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/swav_${RUN_TAG}_seed${SEED}.log}"
TB_LOG_DIR="${TB_LOG_DIR:-${SAVE_DIR}/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

echo "Launching SwAV training with nohup on GPU ${GPU_ID}..."
echo "  batch_size=${BATCH_SIZE}  base_lr=${LR}  effective_lr=$(python -c "print(${LR} * ${BATCH_SIZE} / 256)")"
echo "  epochs=${EPOCHS}  num_prototypes=${NUM_PROTOTYPES}  seed=${SEED}"
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python -u pretrain_swav_stl10.py \
  --data-root ./data \
  --save-dir "${SAVE_DIR}" \
  --download \
  --seed "${SEED}" \
  --gpu-id "${GPU_ID}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size 96 \
  --projection-hidden-dim 2048 \
  --projection-out-dim 128 \
  --num-prototypes "${NUM_PROTOTYPES}" \
  --temperature 0.1 \
  --sinkhorn-epsilon 0.05 \
  --sinkhorn-iterations 3 \
  --assignment-crop-ids 0 1 \
  --lr "${LR}" \
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

echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT} --host 0.0.0.0"
