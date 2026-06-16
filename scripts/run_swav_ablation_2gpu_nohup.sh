#!/usr/bin/env bash
set -euo pipefail

# 2-run SwAV ablation launcher (GPU 0 and 1)
# Baseline family: pretrain_swav_stl10.py
# - GPU0: num_prototypes=1024, lr=0.2
# - GPU1: num_prototypes=512,  lr=0.15

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-swav_ab2g}"

launch_run() {
  local gpu="$1"
  local run_name="$2"
  local num_prototypes="$3"
  local lr="$4"

  local save_dir="./outputs/${run_name}"
  local log_file="./logs/${run_name}.log"

  echo "[launch] gpu=${gpu} run=${run_name} num_prototypes=${num_prototypes} lr=${lr}"
  nohup python pretrain_swav_stl10.py \
    --data-root "${DATA_ROOT}" \
    --save-dir "${save_dir}" \
    --download \
    --seed "${SEED}" \
    --gpu-id "${gpu}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --image-size 96 \
    --projection-hidden-dim 2048 \
    --projection-out-dim 128 \
    --num-prototypes "${num_prototypes}" \
    --temperature 0.1 \
    --sinkhorn-epsilon 0.05 \
    --sinkhorn-iterations 3 \
    --assignment-crop-ids 0 1 \
    --lr "${lr}" \
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
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
}

launch_run 0 "${RUN_TAG}_g0_proto1024_lr02_seed${SEED}_e${EPOCHS}" 1024 0.2
launch_run 1 "${RUN_TAG}_g1_proto512_lr015_seed${SEED}_e${EPOCHS}" 512 0.15

echo
echo "Launched 2 SwAV ablation runs on GPU 0 and 1."
echo "Logs: ls -1 ./logs/${RUN_TAG}_*.log"
echo "PIDs: ls -1 ./logs/${RUN_TAG}_*.log.pid"
echo "Monitor: nvidia-smi"
