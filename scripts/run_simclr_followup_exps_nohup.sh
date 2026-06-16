#!/usr/bin/env bash
# SimCLR follow-up experiments on GPU 0-3 (one experiment per GPU, nohup background).
#
#   GPU 0 — Exp1: exp4 recipe + weight_decay 5e-5, e500
#   GPU 1 — Exp2: temperature sweep @ e500 (default temp=0.18, override with TEMP_EXP2)
#   GPU 2 — Exp4: exp4 saturation check, e1000
#   GPU 3 — Exp5: stronger aug + warmup 20, e500
#
# Usage:
#   bash scripts/run_simclr_followup_exps_nohup.sh
#   EPOCHS_SAT=600 bash scripts/run_simclr_followup_exps_nohup.sh   # shorter saturation run
#   TEMP_EXP2=0.22 bash scripts/run_simclr_followup_exps_nohup.sh   # exp2 temp override
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs

SEED="${SEED:-42}"
EPOCHS_LONG="${EPOCHS_LONG:-500}"
EPOCHS_SAT="${EPOCHS_SAT:-1000}"
TEMP_EXP2="${TEMP_EXP2:-0.18}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
RUN_TAG="${RUN_TAG:-simclr_followup}"
SAVE_EVERY="${SAVE_EVERY:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

GPU_EXP1="${GPU_EXP1:-0}"
GPU_EXP2="${GPU_EXP2:-1}"
GPU_EXP4="${GPU_EXP4:-2}"
GPU_EXP5="${GPU_EXP5:-3}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --download
  --seed "${SEED}"
  --batch-size 320
  --num-workers "${NUM_WORKERS}"
  --image-size 96
  --projection-hidden-dim 2048
  --projection-out-dim 256
  --lr 0.3
  --momentum 0.9
  --warmup-epochs 10
  --color-jitter-strength 0.5
  --gaussian-blur-prob 0.5
  --amp
  --tensorboard
  --eval-interval "${EVAL_INTERVAL}"
  --eval-batch-size 256
  --linear-eval-epochs 100
  --linear-eval-lr 0.01
  --knn-k 20
  --knn-temperature 0.1
  --periodic-keep-last 3
  --log-interval 20
  --save-every "${SAVE_EVERY}"
  --disable-tqdm
)

launch_run() {
  local gpu="$1"
  local run_name="$2"
  local epochs="$3"
  shift 3
  local extra_args=("$@")

  local save_dir="./outputs/${run_name}"
  local log_file="./logs/${run_name}.log"

  echo "[launch] gpu=${gpu} epochs=${epochs} run=${run_name}"
  nohup python -u pretrain_simclr_stl10.py \
    "${COMMON_ARGS[@]}" \
    --save-dir "${save_dir}" \
    --gpu-id "${gpu}" \
    --epochs "${epochs}" \
    --temperature 0.2 \
    --weight-decay 0.0001 \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  echo "  pid=${pid} log=${log_file} save_dir=${save_dir}"
}

TEMP_TAG="${TEMP_EXP2//./}"

echo "============================================================"
echo "SimCLR follow-up: 4 runs on GPU ${GPU_EXP1},${GPU_EXP2},${GPU_EXP4},${GPU_EXP5}"
echo "============================================================"

# Exp1 — GPU 0: wd 5e-5, e500
launch_run "${GPU_EXP1}" \
  "${RUN_TAG}_exp1_wd5e5_seed${SEED}_e${EPOCHS_LONG}" \
  "${EPOCHS_LONG}" \
  --weight-decay 0.00005

# Exp2 — GPU 1: temperature @ e500
launch_run "${GPU_EXP2}" \
  "${RUN_TAG}_exp2_temp${TEMP_TAG}_seed${SEED}_e${EPOCHS_LONG}" \
  "${EPOCHS_LONG}" \
  --temperature "${TEMP_EXP2}"

# Exp4 — GPU 2: saturation e1000 (exp4 baseline)
launch_run "${GPU_EXP4}" \
  "${RUN_TAG}_exp4_saturation_seed${SEED}_e${EPOCHS_SAT}" \
  "${EPOCHS_SAT}"

# Exp5 — GPU 3: aug + warmup
launch_run "${GPU_EXP5}" \
  "${RUN_TAG}_exp5_aug_warmup_seed${SEED}_e${EPOCHS_LONG}" \
  "${EPOCHS_LONG}" \
  --warmup-epochs 20 \
  --color-jitter-strength 0.6 \
  --gaussian-blur-prob 0.6

echo
echo "All 4 runs launched in background (nohup)."
echo "  GPU ${GPU_EXP1}: Exp1  wd=5e-5,           e${EPOCHS_LONG}"
echo "  GPU ${GPU_EXP2}: Exp2  temp=${TEMP_EXP2},       e${EPOCHS_LONG}"
echo "  GPU ${GPU_EXP4}: Exp4  saturation,        e${EPOCHS_SAT}"
echo "  GPU ${GPU_EXP5}: Exp5  aug+warmup20,      e${EPOCHS_LONG}"
echo
echo "Logs:  ls -1 ./logs/${RUN_TAG}_*.log"
echo "PIDs:  ls -1 ./logs/${RUN_TAG}_*.log.pid"
echo "Tail:  tail -f ./logs/${RUN_TAG}_exp4_saturation_seed${SEED}_e${EPOCHS_SAT}.log"
echo "GPU:   nvidia-smi"
