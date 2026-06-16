from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainState:
    epoch: int
    best_loss: float
    global_step: int


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    state: TrainState,
    is_best: bool = False,
) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": asdict(state),
    }

    last_path = out_dir / "checkpoint_last.pt"
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = out_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)


def save_backbone_weights(save_dir: str, model: torch.nn.Module, epoch: int) -> str:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backbone_epoch_{epoch:04d}.pt"
    torch.save(model.backbone.state_dict(), path)
    return str(path)


def dump_config(save_dir: str, config: dict[str, Any]) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def setup_cuda(single_gpu_id: int = 0) -> torch.device:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(single_gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this project requires single-GPU training.")
    return torch.device("cuda")
