#!/usr/bin/env bash
set -euo pipefail

# 7-GPU ablation launcher for SwAV-MCQ (GPU 0,1,2,3,4,6,7)
# Safe-memory profile: fixed lower-memory base and ablate mostly memory-light params.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-swavmcq_ab7g_safe}"
BATCH_SIZE="${BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-192}"

# Help reduce allocator fragmentation on long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch_run() {
  local gpu="$1"
  local run_name="$2"
  local extra_args="$3"

  local save_dir="./outputs/${run_name}"
  local log_file="./logs/${run_name}.log"

  echo "[launch] gpu=${gpu} run=${run_name}"
  nohup python pretrain_swav_stl10_mcq.py \
    --data-root "${DATA_ROOT}" \
    --save-dir "${save_dir}" \
    --download \
    --seed "${SEED}" \
    --gpu-id "${gpu}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --image-size 96 \
    --local-crop-size 64 \
    --local-crops-number 4 \
    --global-crops-scale 0.4 1.0 \
    --local-crops-scale 0.05 0.4 \
    --projection-hidden-dim 2048 \
    --projection-out-dim 128 \
    --num-prototypes 3000 \
    --temperature 0.1 \
    --sinkhorn-epsilon 0.05 \
    --sinkhorn-iterations 3 \
    --assignment-crop-ids 0 1 \
    --queue-size 2048 \
    --queue-start-epoch 15 \
    --optimizer lars \
    --base-lr 0.3 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --warmup-epochs 10 \
    --amp \
    --tensorboard \
    --eval-interval 25 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --linear-eval-epochs 10 \
    --linear-eval-lr 0.1 \
    --knn-k 20 \
    --knn-temperature 0.1 \
    --periodic-keep-last 3 \
    --log-interval 20 \
    --save-every 25 \
    --disable-tqdm \
    ${extra_args} \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
}

# GPU 0: Baseline (safe memory)
launch_run 0 "${RUN_TAG}_g0_baseline_seed${SEED}_e${EPOCHS}" ""

# GPU 1: Higher temperature
launch_run 1 "${RUN_TAG}_g1_temp02_seed${SEED}_e${EPOCHS}" \
  "--temperature 0.2"

# GPU 2: No queue
launch_run 2 "${RUN_TAG}_g2_noqueue_seed${SEED}_e${EPOCHS}" \
  "--queue-size 0"

# GPU 3: Larger queue
launch_run 3 "${RUN_TAG}_g3_q4096_seed${SEED}_e${EPOCHS}" \
  "--queue-size 4096"

# GPU 4: Earlier queue start
launch_run 4 "${RUN_TAG}_g4_qstart5_seed${SEED}_e${EPOCHS}" \
  "--queue-start-epoch 5"

# GPU 6: Higher LR scaling base
launch_run 6 "${RUN_TAG}_g6_blr045_seed${SEED}_e${EPOCHS}" \
  "--base-lr 0.45"

# GPU 7: Lower sinkhorn epsilon
launch_run 7 "${RUN_TAG}_g7_eps003_seed${SEED}_e${EPOCHS}" \
  "--sinkhorn-epsilon 0.03"

echo
echo "Launched 7 SwAV-MCQ ablation runs on GPU 0,1,2,3,4,6,7 (GPU 5 excluded)."
echo "Log files: ls -1 ./logs/${RUN_TAG}_*.log"
echo "PIDs:      ls -1 ./logs/${RUN_TAG}_*.log.pid"
echo "Monitor:   nvidia-smi"
