#!/usr/bin/env bash
set -euo pipefail

# 8-GPU ablation launcher (single-process per GPU).
# Grid:
# - batch_size:         [256, 320]
# - temperature:        [0.2, 0.5]
# - projection_out_dim: [128, 256]
#
# Keep everything else aligned with your current baseline setting.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-ab8g}"
BASE_LR="${BASE_LR:-0.3}"

GPU_IDS=(0 1 2 3 4 5 6 7)
BATCHES=(256 320)
TEMPS=(0.2 0.5)
OUT_DIMS=(128 256)

run_idx=0
for bs in "${BATCHES[@]}"; do
  for temp in "${TEMPS[@]}"; do
    for out_dim in "${OUT_DIMS[@]}"; do
      gpu="${GPU_IDS[$run_idx]}"

      run_name="${RUN_TAG}_seed${SEED}_e${EPOCHS}_bs${bs}_t${temp}_out${out_dim}_gpu${gpu}"
      save_dir="./outputs/${run_name}"
      log_file="./logs/${run_name}.log"

      echo "[launch] gpu=${gpu} run=${run_name} base_lr=${BASE_LR}"
      nohup \
        python pretrain_simclr_stl10.py \
          --data-root "${DATA_ROOT}" \
          --save-dir "${save_dir}" \
          --download \
          --seed "${SEED}" \
          --gpu-id "${gpu}" \
          --epochs "${EPOCHS}" \
          --batch-size "${bs}" \
          --num-workers "${NUM_WORKERS}" \
          --image-size 96 \
          --temperature "${temp}" \
          --projection-hidden-dim 2048 \
          --projection-out-dim "${out_dim}" \
          --lr "${BASE_LR}" \
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

      pid=$!
      echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
      run_idx=$((run_idx + 1))
    done
  done
done

echo
echo "Launched 8 ablation runs."
echo "Check logs: ls -1 ./logs/${RUN_TAG}_seed${SEED}_e${EPOCHS}_*.log"
echo "Check processes: nvidia-smi"
