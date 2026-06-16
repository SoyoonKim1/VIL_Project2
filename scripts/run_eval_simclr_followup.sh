#!/usr/bin/env bash
# Downstream linear probing (evaluation.py) for SimCLR follow-up runs.
#
# Default targets (RUN_TAG=simclr_followup, SEED=42):
#   simclr_followup_exp1_wd5e5_seed42_e500
#   simclr_followup_exp2_temp018_seed42_e500
#   simclr_followup_exp4_saturation_seed42_e1000
#   simclr_followup_exp5_aug_warmup_seed42_e500
#
# Usage:
#   bash scripts/run_eval_simclr_followup.sh              # all runs
#   bash scripts/run_eval_simclr_followup.sh exp1         # one run
#   CKPT_PATH=./outputs/.../backbone_best.pt \
#     RUN_NAME=my_run bash scripts/run_eval_simclr_followup.sh single
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
EPOCHS_LONG="${EPOCHS_LONG:-500}"
EPOCHS_SAT="${EPOCHS_SAT:-1000}"
TEMP_EXP2="${TEMP_EXP2:-0.18}"
RUN_TAG="${RUN_TAG:-simclr_followup}"
TEMP_TAG="${TEMP_EXP2//./}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-./data}"
FEATURE_ROOT="${FEATURE_ROOT:-./outputs/eval_features}"

RUNS=(
  "${RUN_TAG}_exp1_wd5e5_seed${SEED}_e${EPOCHS_LONG}"
  "${RUN_TAG}_exp2_temp${TEMP_TAG}_seed${SEED}_e${EPOCHS_LONG}"
  "${RUN_TAG}_exp4_saturation_seed${SEED}_e${EPOCHS_SAT}"
  "${RUN_TAG}_exp5_aug_warmup_seed${SEED}_e${EPOCHS_LONG}"
)

eval_one() {
  local run_name="$1"
  local ckpt_path="${2:-./outputs/${run_name}/backbone_best.pt}"
  local out_dir="${FEATURE_ROOT}/${run_name}"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] checkpoint not found: ${ckpt_path}"
    return 1
  fi

  mkdir -p "${out_dir}" logs
  local log_file="./logs/eval_${run_name}.log"

  echo "============================================================"
  echo "[eval] run=${run_name}"
  echo "  checkpoint=${ckpt_path}"
  echo "  features=${out_dir}"
  echo "  log=${log_file}"
  echo "============================================================"

  CKPT_PATH="${ckpt_path}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DATA_ROOT="${DATA_ROOT}" \
  python - <<'PY'
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import STL10, CIFAR10
from simclr.model import SimCLRModel

ckpt = os.environ["CKPT_PATH"]
out_dir = os.environ["OUT_DIR"]
device_str = os.environ["DEVICE"]
num_workers = int(os.environ["NUM_WORKERS"])
data_root = os.environ["DATA_ROOT"]
os.makedirs(out_dir, exist_ok=True)

device = torch.device(device_str if torch.cuda.is_available() else "cpu")
print("device:", device)

transform = T.Compose([
    T.Resize(112, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(96),
    T.ToTensor(),
    T.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713)),
])

# projection head is unused; only backbone weights are loaded.
model = SimCLRModel(backbone_name="resnet50", projection_hidden_dim=2048, projection_out_dim=256)
state = torch.load(ckpt, map_location="cpu")
model.backbone.load_state_dict(state, strict=True)
model = model.to(device)
model.eval()

@torch.no_grad()
def extract(loader):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats.append(model.forward_backbone(x).cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

jobs = [
    ("stl10_train", STL10(root=data_root, split="train", transform=transform, download=True)),
    ("stl10_test", STL10(root=data_root, split="test", transform=transform, download=True)),
    ("cifar10_train", CIFAR10(root=data_root, train=True, transform=transform, download=True)),
    ("cifar10_test", CIFAR10(root=data_root, train=False, transform=transform, download=True)),
]

for name, ds in jobs:
    print("extracting", name, "N=", len(ds))
    loader = DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    f, y = extract(loader)
    np.save(os.path.join(out_dir, f"{name}_features.npy"), f)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), y)
    print("saved", name, f.shape, y.shape)
PY

  python evaluation.py \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --data-root "${DATA_ROOT}" \
    --download \
    --stl10-train-features "${out_dir}/stl10_train_features.npy" \
    --stl10-train-labels "${out_dir}/stl10_train_labels.npy" \
    --stl10-test-features "${out_dir}/stl10_test_features.npy" \
    --stl10-test-labels "${out_dir}/stl10_test_labels.npy" \
    --cifar10-train-features "${out_dir}/cifar10_train_features.npy" \
    --cifar10-train-labels "${out_dir}/cifar10_train_labels.npy" \
    --cifar10-test-features "${out_dir}/cifar10_test_features.npy" \
    --cifar10-test-labels "${out_dir}/cifar10_test_labels.npy" \
    2>&1 | tee "${log_file}"

  echo "[done] ${run_name}"
}

TARGET="${1:-all}"
case "${TARGET}" in
  all)
    for run in "${RUNS[@]}"; do
      eval_one "${run}" || true
    done
    ;;
  exp1) eval_one "${RUNS[0]}" ;;
  exp2) eval_one "${RUNS[1]}" ;;
  exp4) eval_one "${RUNS[2]}" ;;
  exp5) eval_one "${RUNS[3]}" ;;
  single)
    RUN_NAME="${RUN_NAME:?Set RUN_NAME for single mode}"
    CKPT_PATH="${CKPT_PATH:-./outputs/${RUN_NAME}/backbone_best.pt}"
    eval_one "${RUN_NAME}" "${CKPT_PATH}"
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    echo "Use: all | exp1 | exp2 | exp4 | exp5 | single"
    exit 1
    ;;
esac

echo
echo "Evaluation logs: ls -1 ./logs/eval_${RUN_TAG}_*.log"
