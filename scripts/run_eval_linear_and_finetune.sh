#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT_PATH="${CKPT_PATH:-./outputs/ab8g_v2_seed42_e200_bs320_t0.2_out256_gpu5/backbone_best.pt}"
RUN_NAME="${RUN_NAME:-ab8g_v2_seed42_e200_bs320_t0.2_out256_gpu5}"
OUT_DIR="${OUT_DIR:-./outputs/eval_features/${RUN_NAME}}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"

export CKPT_PATH
export OUT_DIR
export DEVICE
export NUM_WORKERS
export DATA_ROOT

mkdir -p "${OUT_DIR}" logs

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

echo "[1/3] Extract features for STL10/CIFAR10"
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

model = SimCLRModel(backbone_name="resnet50", projection_hidden_dim=2048, projection_out_dim=128)
state = torch.load(ckpt, map_location="cpu")
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
    ("stl10_train", STL10(root=data_root, split="train", transform=transform, download=True)),
    ("stl10_test", STL10(root=data_root, split="test", transform=transform, download=True)),
    ("cifar10_train", CIFAR10(root=data_root, train=True, transform=transform, download=True)),
    ("cifar10_test", CIFAR10(root=data_root, train=False, transform=transform, download=True)),
]

for name, ds in jobs:
    print("extracting", name, "N=", len(ds))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    f, y = extract(loader)
    np.save(os.path.join(out_dir, f"{name}_features.npy"), f)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), y)
    print("saved", name, f.shape, y.shape)
PY

echo "[2/3] Fixed linear probing (STL10)"
python evaluation.py \
  --dataset stl10 \
  --data-root "${DATA_ROOT}" \
  --download \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --train-features "${OUT_DIR}/stl10_train_features.npy" \
  --test-features "${OUT_DIR}/stl10_test_features.npy"

echo "[2/3] Fixed linear probing (CIFAR10)"
python evaluation.py \
  --dataset cifar10 \
  --data-root "${DATA_ROOT}" \
  --download \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --train-features "${OUT_DIR}/cifar10_train_features.npy" \
  --test-features "${OUT_DIR}/cifar10_test_features.npy"

echo "[3/3] Fixed fine-tuning (STL10 + CIFAR10)"
python evaluation_finetune.py \
  --checkpoint "${CKPT_PATH}" \
  --dataset both \
  --data-root "${DATA_ROOT}" \
  --download \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}"

echo "[done] linear probe + fine-tuning evaluation finished"
