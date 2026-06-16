#!/usr/bin/env bash
set -euo pipefail

# 5-run SimCLR ablation launcher (LARS excluded).
# Runs are pinned one-per-GPU using default GPU ids: 1,2,3,4,5.
# Baseline comes from: outputs/ab8g_v2_seed42_e200_bs320_t0.2_out256_gpu5/train_config.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-ab5g_simclr_no_lars}"
SAVE_EVERY="${SAVE_EVERY:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

# Common fixed baseline values
BASE_EPOCHS="${BASE_EPOCHS:-200}"
BASE_BATCH="${BASE_BATCH:-320}"
BASE_TEMP="${BASE_TEMP:-0.2}"
BASE_HID="${BASE_HID:-2048}"
BASE_OUT="${BASE_OUT:-256}"
BASE_LR="${BASE_LR:-0.3}"
BASE_MOMENTUM="${BASE_MOMENTUM:-0.9}"
BASE_WD="${BASE_WD:-0.0001}"
BASE_WARMUP="${BASE_WARMUP:-10}"

# Requested common eval setting for all runs
LINEAR_EVAL_EPOCHS="${LINEAR_EVAL_EPOCHS:-100}"
LINEAR_EVAL_LR="${LINEAR_EVAL_LR:-0.01}"

# GPU mapping (override if needed)
GPU1="${GPU1:-1}"  # Exp1 baseline
GPU2="${GPU2:-2}"  # Exp2 temperature 0.1
GPU3="${GPU3:-3}"  # Exp3 projection out dim 128
GPU4="${GPU4:-4}"  # Exp4 epochs 400
GPU6="${GPU6:-5}"  # Exp6 batch 512 (name kept as Exp6)

launch_run() {
  local gpu="$1"
  local run_name="$2"
  local extra_args="$3"

  local save_dir="./outputs/${run_name}"
  local log_file="./logs/${run_name}.log"

  echo "[launch] gpu=${gpu} run=${run_name}"
  nohup python -u pretrain_simclr_stl10.py     --data-root "${DATA_ROOT}"     --save-dir "${save_dir}"     --download     --seed "${SEED}"     --gpu-id "${gpu}"     --epochs "${BASE_EPOCHS}"     --batch-size "${BASE_BATCH}"     --num-workers "${NUM_WORKERS}"     --image-size 96     --temperature "${BASE_TEMP}"     --projection-hidden-dim "${BASE_HID}"     --projection-out-dim "${BASE_OUT}"     --lr "${BASE_LR}"     --momentum "${BASE_MOMENTUM}"     --weight-decay "${BASE_WD}"     --warmup-epochs "${BASE_WARMUP}"     --amp     --tensorboard     --eval-interval "${EVAL_INTERVAL}"     --eval-batch-size 256     --linear-eval-epochs "${LINEAR_EVAL_EPOCHS}"     --linear-eval-lr "${LINEAR_EVAL_LR}"     --knn-k 20     --knn-temperature 0.1     --periodic-keep-last 3     --log-interval 20     --save-every "${SAVE_EVERY}"     --disable-tqdm     ${extra_args}     > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
}

# Exp1 — Baseline
launch_run "${GPU1}" "${RUN_TAG}_exp1_baseline_seed${SEED}_e${BASE_EPOCHS}"   ""

# Exp2 — Temperature: 0.2 -> 0.1
launch_run "${GPU2}" "${RUN_TAG}_exp2_temp01_seed${SEED}_e${BASE_EPOCHS}"   "--temperature 0.1"

# Exp3 — Projection out dim: 256 -> 128
launch_run "${GPU3}" "${RUN_TAG}_exp3_out128_seed${SEED}_e${BASE_EPOCHS}"   "--projection-out-dim 128"

# Exp4 — Epochs: 200 -> 400
launch_run "${GPU4}" "${RUN_TAG}_exp4_e400_seed${SEED}_e400"   "--epochs 400"

# Exp6 — Batch size: 320 -> 512 with linear LR scaling from baseline
# lr_scaled = 0.3 * (512/320) = 0.48
launch_run "${GPU6}" "${RUN_TAG}_exp6_bs512_lr048_seed${SEED}_e${BASE_EPOCHS}"   "--batch-size 512 --lr 0.48"

echo
echo "Launched 5 SimCLR ablation runs (LARS excluded)."
echo "Logs: ls -1 ./logs/${RUN_TAG}_*.log"
echo "PIDs: ls -1 ./logs/${RUN_TAG}_*.log.pid"
echo "Monitor: nvidia-smi"
