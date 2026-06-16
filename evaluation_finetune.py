"""
Fixed fine-tuning evaluator (no KNN) for SSL backbones.

Protocol:
  - Dataset:
      STL10   -> train split (5k), test split (8k)
      CIFAR10 -> train split (50k), test split (10k)
  - Model:
      torchvision ResNet-50 + Linear(num_classes) head
      initialize backbone from checkpoint
  - Optimization:
      SGD(lr=0.1, momentum=0.9, weight_decay=0.0)
      cosine annealing, 100 epochs, batch size 128
  - Evaluation:
      final epoch test Top-1, no validation/model selection

Do not modify hyperparameters for fair comparison.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10
from torchvision.models import resnet50

# Fixed hyperparameters (DO NOT change)
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed fine-tuning evaluator (no KNN).")
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint path.")
    parser.add_argument("--dataset", choices=["stl10", "cifar10", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Seed for fine-tuning evaluator only.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_backbone_state_dict(loaded_obj: object) -> dict[str, torch.Tensor]:
    if isinstance(loaded_obj, dict) and "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
        state = loaded_obj["model"]
    elif isinstance(loaded_obj, dict):
        state = loaded_obj
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    keys = list(state.keys())
    if not keys:
        raise RuntimeError("Checkpoint state dict is empty.")

    if any(k.startswith("backbone.") for k in keys):
        return {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
    return {k: v for k, v in state.items()}


def _build_dataset(dataset: str, data_root: str, download: bool):
    if dataset == "stl10":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4467, 0.4398, 0.4066),
                    std=(0.2603, 0.2566, 0.2713),
                ),
            ]
        )
        train_ds = STL10(root=data_root, split="train", transform=transform, download=download)
        test_ds = STL10(root=data_root, split="test", transform=transform, download=download)
    elif dataset == "cifar10":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        train_ds = CIFAR10(root=data_root, train=True, transform=transform, download=download)
        test_ds = CIFAR10(root=data_root, train=False, transform=transform, download=download)
    else:
        raise RuntimeError(f"Unsupported dataset: {dataset}")
    return train_ds, test_ds


def _dataset_num_classes(dataset: str) -> int:
    if dataset in ("stl10", "cifar10"):
        return 10
    raise RuntimeError(f"Unsupported dataset: {dataset}")


def run_finetune(
    *,
    dataset: str,
    checkpoint_path: str,
    data_root: str,
    device: torch.device,
    num_workers: int,
    download: bool,
) -> float:
    train_ds, test_ds = _build_dataset(dataset, data_root=data_root, download=download)
    num_classes = _dataset_num_classes(dataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = resnet50(weights=None)
    in_dim = model.fc.in_features
    model.fc = torch.nn.Linear(in_dim, num_classes)

    loaded = torch.load(checkpoint_path, map_location="cpu")
    backbone_state = _extract_backbone_state_dict(loaded)

    # Ignore classifier weights from checkpoint (if any).
    backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("fc.")]
    missing = [k for k in missing if not k.startswith("fc.")]
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing backbone keys while loading checkpoint: {missing}")

    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        scheduler.step()
        print(
            f"[{dataset}] epoch {epoch + 1:03d}/{EPOCHS} "
            f"loss={total_loss / max(total, 1):.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(images).argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / max(total, 1)
    print(f"[{dataset}] Fine-tuning Top-1: {acc:.2f}%")
    return acc


def _iter_datasets(dataset_arg: str) -> Iterable[str]:
    if dataset_arg == "both":
        return ("stl10", "cifar10")
    return (dataset_arg,)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        raise RuntimeError(f"Checkpoint not found: {args.checkpoint}")

    print("Fixed fine-tuning recipe:")
    print(f"  SGD lr={LR} momentum={MOMENTUM} weight_decay={WEIGHT_DECAY}")
    print(f"  cosine annealing, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    print("  final epoch evaluation, no validation/model selection")
    print(f"  device={device}")

    results: dict[str, float] = {}
    for ds_name in _iter_datasets(args.dataset):
        print(f"\n== {ds_name} ==")
        results[ds_name] = run_finetune(
            dataset=ds_name,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            device=device,
            num_workers=args.num_workers,
            download=args.download,
        )

    print("\nFinal Results (Fine-tuning)")
    for name, acc in results.items():
        print(f"{name:10s} Top-1: {acc:.2f}%")


if __name__ == "__main__":
    main()
