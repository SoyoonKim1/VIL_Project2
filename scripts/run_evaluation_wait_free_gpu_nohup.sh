#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs/eval_features

# Override with env vars if needed.
CKPT_PATH="${CKPT_PATH:-./outputs/ab8g_v2_seed42_e200_bs320_t0.2_out256_gpu5/backbone_best.pt}"
RUN_NAME="${RUN_NAME:-ab8g_v2_seed42_e200_bs320_t0.2_out256_gpu5}"
FEATURE_OUT_DIR="${FEATURE_OUT_DIR:-./outputs/eval_features/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-./logs/eval_${RUN_NAME}.log}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

wait_for_free_gpu() {
  local candidates_csv="$1"
  local poll_seconds="$2"

  while true; do
    while IFS=, read -r idx mem util; do
      idx="$(echo "$idx" | xargs)"
      mem="$(echo "$mem" | xargs)"
      util="$(echo "$util" | xargs)"

      case ",${candidates_csv}," in
        *",${idx},"*) ;;
        *) continue ;;
      esac

      # Conservative free criteria.
      if [[ "${mem}" -lt 1024 && "${util}" -lt 20 ]]; then
        echo "${idx}"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

    echo "[wait] no free GPU yet; sleep ${poll_seconds}s"
    sleep "${poll_seconds}"
  done
}

echo "Starting evaluation launcher in background..."
echo "  checkpoint: ${CKPT_PATH}"
echo "  run_name:   ${RUN_NAME}"
echo "  log_file:   ${LOG_FILE}"
echo "  candidates: ${GPU_CANDIDATES}"

nohup bash -lc "
set -euo pipefail
cd \"${ROOT_DIR}\"
echo \"[info] waiting for free GPU among: ${GPU_CANDIDATES}\"

FREE_GPU=''
while true; do
  while IFS=, read -r idx mem util; do
    idx=\$(echo \"\$idx\" | xargs)
    mem=\$(echo \"\$mem\" | xargs)
    util=\$(echo \"\$util\" | xargs)
    case \",${GPU_CANDIDATES},\" in
      *\",\$idx,\"*) ;;
      *) continue ;;
    esac
    if [[ \"\$mem\" -lt 1024 && \"\$util\" -lt 20 ]]; then
      FREE_GPU=\"\$idx\"
      break 2
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  echo \"[wait] no free GPU yet; sleep ${POLL_SECONDS}s\"
  sleep \"${POLL_SECONDS}\"
done

echo \"[info] selected GPU: \${FREE_GPU}\"
mkdir -p \"${FEATURE_OUT_DIR}\"
export FREE_GPU

python - <<'PY'
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import STL10, CIFAR10
from simclr.model import SimCLRModel

ckpt = '${CKPT_PATH}'
out_dir = '${FEATURE_OUT_DIR}'
gpu_id = os.environ['FREE_GPU']
num_workers = ${NUM_WORKERS}
os.makedirs(out_dir, exist_ok=True)

device = torch.device(f'cuda:{gpu_id}')
print('device:', device)

transform = T.Compose([
    T.Resize(112, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(96),
    T.ToTensor(),
    T.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713)),
])

model = SimCLRModel(backbone_name='resnet50', projection_hidden_dim=2048, projection_out_dim=128)
state = torch.load(ckpt, map_location='cpu')
model.backbone.load_state_dict(state, strict=True)
model = model.to(device)
model.eval()

@torch.no_grad()
def extract(loader):
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.forward_backbone(x)
        feats.append(f.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

jobs = [
    ('stl10_train', STL10(root='./data', split='train', transform=transform, download=True)),
    ('stl10_test', STL10(root='./data', split='test', transform=transform, download=True)),
    ('cifar10_train', CIFAR10(root='./data', train=True, transform=transform, download=True)),
    ('cifar10_test', CIFAR10(root='./data', train=False, transform=transform, download=True)),
]

for name, ds in jobs:
    print('extracting', name, 'N=', len(ds))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    f, y = extract(loader)
    np.save(os.path.join(out_dir, f'{name}_features.npy'), f)
    np.save(os.path.join(out_dir, f'{name}_labels.npy'), y)
    print('saved', name, f.shape, y.shape)
PY

python evaluation.py \
  --dataset stl10 \
  --data-root ./data \
  --download \
  --device \"cuda:\${FREE_GPU}\" \
  --num-workers \"${NUM_WORKERS}\" \
  --train-features \"${FEATURE_OUT_DIR}/stl10_train_features.npy\" \
  --test-features \"${FEATURE_OUT_DIR}/stl10_test_features.npy\"

python evaluation.py \
  --dataset cifar10 \
  --data-root ./data \
  --download \
  --device \"cuda:\${FREE_GPU}\" \
  --num-workers \"${NUM_WORKERS}\" \
  --train-features \"${FEATURE_OUT_DIR}/cifar10_train_features.npy\" \
  --test-features \"${FEATURE_OUT_DIR}/cifar10_test_features.npy\"

echo \"[done] evaluation finished\"
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_FILE}.pid"
echo "Started waiting job. PID=${PID}"
echo "PID file: ${LOG_FILE}.pid"
echo "Tail log: tail -f ${LOG_FILE}"
