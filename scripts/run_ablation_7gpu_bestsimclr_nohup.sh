#!/usr/bin/env bash
set -euo pipefail

# 7-GPU ablation launcher based on your best SimCLR run:
#   bs=320, temp=0.2, projection_out_dim=256
# Runs are pinned to GPU 1..7 (one run per GPU).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-ab7g_bestsimclr}"
BASE_LR="${BASE_LR:-0.3}"
SAVE_EVERY="${SAVE_EVERY:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

launch_run() {
  local gpu="$1"
  local run_name="$2"
  local extra_args="$3"

  local save_dir="./outputs/${run_name}"
  local log_file="./logs/${run_name}.log"

  echo "[launch] gpu=${gpu} run=${run_name}"
  nohup python pretrain_simclr_stl10.py \
    --data-root "${DATA_ROOT}" \
    --save-dir "${save_dir}" \
    --download \
    --seed "${SEED}" \
    --gpu-id "${gpu}" \
    --epochs "${EPOCHS}" \
    --batch-size 320 \
    --num-workers "${NUM_WORKERS}" \
    --image-size 96 \
    --temperature 0.2 \
    --projection-hidden-dim 2048 \
    --projection-out-dim 256 \
    --lr "${BASE_LR}" \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --warmup-epochs 10 \
    --amp \
    --tensorboard \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-batch-size 256 \
    --linear-eval-epochs 10 \
    --linear-eval-lr 0.1 \
    --knn-k 20 \
    --knn-temperature 0.1 \
    --periodic-keep-last 3 \
    --log-interval 20 \
    --save-every "${SAVE_EVERY}" \
    --disable-tqdm \
    ${extra_args} \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
}

# GPU1: temperature lower
launch_run 1 "${RUN_TAG}_g1_temp015_seed${SEED}_e${EPOCHS}" \
  "--temperature 0.15"

# GPU2: slightly lower temperature
launch_run 2 "${RUN_TAG}_g2_temp018_seed${SEED}_e${EPOCHS}" \
  "--temperature 0.18"

# GPU3: slightly higher temperature
launch_run 3 "${RUN_TAG}_g3_temp022_seed${SEED}_e${EPOCHS}" \
  "--temperature 0.22"

# GPU4: larger projection output dimension
launch_run 4 "${RUN_TAG}_g4_out384_seed${SEED}_e${EPOCHS}" \
  "--projection-out-dim 384"

# GPU5: larger batch (effective lr scales automatically in training code)
launch_run 5 "${RUN_TAG}_g5_bs384_seed${SEED}_e${EPOCHS}" \
  "--batch-size 384"

# GPU6: lower weight decay
launch_run 6 "${RUN_TAG}_g6_wd5e5_seed${SEED}_e${EPOCHS}" \
  "--weight-decay 0.00005"

# GPU7: wider projection hidden + stronger weight decay
launch_run 7 "${RUN_TAG}_g7_hid4096_wd2e4_seed${SEED}_e${EPOCHS}" \
  "--projection-hidden-dim 4096 --weight-decay 0.0002"

echo
echo "Launched 7 SimCLR ablation runs on GPU 1..7."
echo "Logs: ls -1 ./logs/${RUN_TAG}_*.log"
echo "PIDs: ls -1 ./logs/${RUN_TAG}_*.log.pid"
echo "Monitor: nvidia-smi"
