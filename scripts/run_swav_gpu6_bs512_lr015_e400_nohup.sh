#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_DIR="${SAVE_DIR:-./outputs/swav_stl10_gpu6_bs512_lr015_e400_proto512_seed${SEED}}"
LOG_FILE="${LOG_FILE:-./logs/swav_gpu6_bs512_lr015_e400_proto512_seed${SEED}.log}"
TB_LOG_DIR="${TB_LOG_DIR:-${SAVE_DIR}/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

echo "Launching SwAV training with requested config on GPU 6..."
echo "  save_dir: ${SAVE_DIR}"
echo "  log_file: ${LOG_FILE}"

nohup python -u pretrain_swav_stl10.py   --data-root ./data   --save-dir "${SAVE_DIR}"   --download   --seed "${SEED}"   --gpu-id 6   --epochs 400   --batch-size 512   --num-workers "${NUM_WORKERS}"   --image-size 96   --projection-hidden-dim 2048   --projection-out-dim 128   --num-prototypes 512   --temperature 0.1   --sinkhorn-epsilon 0.05   --sinkhorn-iterations 3   --assignment-crop-ids 0 1   --lr 0.15   --momentum 0.9   --weight-decay 0.0001   --warmup-epochs 10   --amp   --tensorboard   --eval-interval 25   --eval-batch-size 256   --linear-eval-epochs 100   --linear-eval-lr 0.01   --knn-k 20   --knn-temperature 0.1   --periodic-keep-last 3   --log-interval 20   --save-every 25   --disable-tqdm   > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"

echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir ${TB_LOG_DIR} --port ${TB_PORT} --host 0.0.0.0"
