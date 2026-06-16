#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs/eval_features

# Evaluation target runs from:
# - logs/swav_ab2g_g0_proto1024_lr02_seed42_e200.log
# - logs/swav_ab2g_g1_proto512_lr015_seed42_e200.log
RUN_A="${RUN_A:-swav_ab2g_g0_proto1024_lr02_seed42_e200}"
RUN_B="${RUN_B:-swav_ab2g_g1_proto512_lr015_seed42_e200}"
RUNS=("${RUN_A}" "${RUN_B}")

DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
LOG_FILE="${LOG_FILE:-./logs/eval_swav_ab2g_pair.log}"

for run_name in "${RUNS[@]}"; do
  ckpt_path="./outputs/${run_name}/backbone_best.pt"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[error] checkpoint not found: ${ckpt_path}"
    exit 1
  fi
done

echo "Launching pair evaluation with nohup..."
echo "  runs: ${RUN_A}, ${RUN_B}"
echo "  log:  ${LOG_FILE}"

nohup bash -lc "
set -euo pipefail
cd \"${ROOT_DIR}\"

echo '============================================================'
echo '[pair-eval] Start: SwAV ablation-2gpu checkpoints'
echo '[pair-eval] Fixed evaluator: evaluation.py + evaluation_finetune.py'
echo '============================================================'

for run_name in \"${RUN_A}\" \"${RUN_B}\"; do
  ckpt_path=\"./outputs/\${run_name}/backbone_best.pt\"
  pretrain_log=\"./logs/\${run_name}.log\"
  out_dir=\"./outputs/eval_features/\${run_name}\"

  echo
  echo '------------------------------------------------------------'
  echo \"[pair-eval] run=\${run_name}\"
  echo \"[pair-eval] checkpoint=\${ckpt_path}\"
  export PRETRAIN_LOG=\"\${pretrain_log}\"
  if [[ -f \"\${pretrain_log}\" ]]; then
    echo \"[pair-eval] pretrain summary from \${pretrain_log}\"
    python - <<'PY'
import os
import re

log_path = os.environ['PRETRAIN_LOG']
best_loss = None
knn = None
linear = None
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'Best loss=' in line:
            m = re.search(r'Best loss=([0-9.]+)', line)
            if m:
                best_loss = float(m.group(1))
        if 'eval @ epoch 200:' in line:
            m = re.search(r'knn_top1=([0-9.]+)%\\s+linear_top1=([0-9.]+)%', line)
            if m:
                knn = float(m.group(1))
                linear = float(m.group(2))

print(f'  best_loss={best_loss}')
print(f'  epoch200_knn_top1={knn}')
print(f'  epoch200_linear_top1={linear}')
PY
  fi

  export CKPT_PATH=\"\${ckpt_path}\"
  export RUN_NAME=\"\${run_name}\"
  export OUT_DIR=\"\${out_dir}\"
  export DEVICE=\"${DEVICE}\"
  export NUM_WORKERS=\"${NUM_WORKERS}\"
  export SEED=\"${SEED}\"
  export DATA_ROOT=\"${DATA_ROOT}\"

  bash ./scripts/run_eval_linear_and_finetune.sh
done

echo
echo '[pair-eval] Done.'
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"
echo "Started. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
