from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import amp
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import STL10
from tqdm import tqdm

from pretrain_simclr_stl10 import run_representation_eval
from simclr.utils import TrainState, dump_config, save_backbone_weights, save_checkpoint, seed_everything, setup_cuda
from swav import SwAVModel, sinkhorn_knopp


def parse_assignment_crop_ids(crop_ids: Sequence[int]) -> tuple[int, ...]:
    if len(crop_ids) == 0:
        raise ValueError("assignment-crop-ids must not be empty.")
    if len(set(crop_ids)) != len(crop_ids):
        raise ValueError(f"assignment-crop-ids must be unique, got {list(crop_ids)}")
    if any(i < 0 for i in crop_ids):
        raise ValueError(f"assignment-crop-ids must be non-negative, got {list(crop_ids)}")
    return tuple(crop_ids)


class MultiCropTransform:
    def __init__(
        self,
        image_size: int = 96,
        local_crops_number: int = 4,
        local_crop_size: int = 48,
        global_scale: tuple[float, float] = (0.14, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.14),
    ) -> None:
        self.local_crops_number = local_crops_number

        normalize = T.Normalize(
            mean=(0.43, 0.42, 0.39),
            std=(0.27, 0.26, 0.27),
        )
        color = T.ColorJitter(0.8, 0.8, 0.8, 0.2)

        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size,
                    scale=global_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                T.ToTensor(),
                normalize,
            ]
        )

        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    local_crop_size,
                    scale=local_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, x):
        crops = [self.global_transform(x), self.global_transform(x)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(x))
        return crops


def build_stl10_multicrop_loader(
    data_root: str,
    image_size: int,
    local_crop_size: int,
    local_crops_number: int,
    global_crops_scale: tuple[float, float],
    local_crops_scale: tuple[float, float],
    batch_size: int,
    num_workers: int,
    download: bool = False,
) -> DataLoader:
    transform = MultiCropTransform(
        image_size=image_size,
        local_crops_number=local_crops_number,
        local_crop_size=local_crop_size,
        global_scale=global_crops_scale,
        local_scale=local_crops_scale,
    )
    ds = STL10(root=data_root, split="unlabeled", transform=transform, download=download)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def build_stl10_eval_loaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    eval_transform = T.Compose(
        [
            T.Resize(image_size + 16, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.4467, 0.4398, 0.4066),
                std=(0.2603, 0.2566, 0.2713),
            ),
        ]
    )
    train_ds = STL10(root=data_root, split="train", transform=eval_transform, download=download)
    test_ds = STL10(root=data_root, split="test", transform=eval_transform, download=download)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return DataLoader(train_ds, shuffle=False, **common), DataLoader(test_ds, shuffle=False, **common)


class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        eta: float = 0.001,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if p.ndim > 1:
                    grad = grad.add(p, alpha=weight_decay)
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(grad)
                    trust_ratio = torch.where(
                        (p_norm > 0) & (g_norm > 0),
                        eta * p_norm / (g_norm + eps),
                        torch.ones_like(p_norm),
                    )
                    scaled_lr = lr * trust_ratio
                else:
                    scaled_lr = lr

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(momentum).add_(grad)
                p.add_(mu, alpha=-scaled_lr)
        return loss


class SwAVQueue:
    def __init__(self, size: int, emb_dim: int, assignment_crop_ids: tuple[int, ...], device: torch.device):
        self.size = max(0, size)
        self.assignment_crop_ids = assignment_crop_ids
        self.queue: dict[int, torch.Tensor] = {}
        self.ptr: dict[int, int] = {}
        self.filled: dict[int, int] = {}
        if self.size > 0:
            for cid in assignment_crop_ids:
                self.queue[cid] = torch.zeros(self.size, emb_dim, device=device)
                self.ptr[cid] = 0
                self.filled[cid] = 0

    def get(self, crop_id: int) -> torch.Tensor | None:
        if self.size <= 0:
            return None
        filled = self.filled[crop_id]
        if filled <= 0:
            return None
        return self.queue[crop_id][:filled]

    @torch.no_grad()
    def enqueue(self, crop_id: int, emb: torch.Tensor) -> None:
        if self.size <= 0:
            return
        bsz = emb.size(0)
        if bsz >= self.size:
            self.queue[crop_id].copy_(emb[-self.size :])
            self.ptr[crop_id] = 0
            self.filled[crop_id] = self.size
            return

        ptr = self.ptr[crop_id]
        end = ptr + bsz
        if end <= self.size:
            self.queue[crop_id][ptr:end].copy_(emb)
        else:
            first = self.size - ptr
            self.queue[crop_id][ptr:].copy_(emb[:first])
            self.queue[crop_id][: bsz - first].copy_(emb[first:])
        self.ptr[crop_id] = (ptr + bsz) % self.size
        self.filled[crop_id] = min(self.size, self.filled[crop_id] + bsz)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    final_lr: float,
):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=final_lr,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=final_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return scheduler


def swav_loss_with_queue(
    model: SwAVModel,
    crop_embeddings: Sequence[torch.Tensor],
    crop_logits: Sequence[torch.Tensor],
    assignment_crop_ids: tuple[int, ...],
    temperature: float,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
    queue: SwAVQueue | None,
    use_queue: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(crop_logits) < 2:
        raise ValueError("Need at least two crops for SwAV loss.")

    total_loss = torch.zeros((), device=crop_logits[0].device)
    terms = 0

    proto_counts = torch.zeros(crop_logits[0].size(1), device=crop_logits[0].device)

    for assign_id in assignment_crop_ids:
        scores = crop_logits[assign_id].detach()
        if use_queue and queue is not None:
            q_emb = queue.get(assign_id)
            if q_emb is not None:
                with torch.no_grad():
                    model._normalize_prototypes_if_needed()
                    q_scores = model.prototypes(q_emb).detach()
                scores = torch.cat([q_scores, scores], dim=0)

        with torch.no_grad():
            assignments = sinkhorn_knopp(
                scores,
                epsilon=sinkhorn_epsilon,
                iterations=sinkhorn_iterations,
            )
            batch_assign = assignments[-crop_logits[assign_id].size(0) :]
            hard_assign = batch_assign.argmax(dim=1)
            proto_counts.scatter_add_(
                0,
                hard_assign,
                torch.ones_like(hard_assign, dtype=proto_counts.dtype),
            )

        for pred_id, logits in enumerate(crop_logits):
            if pred_id == assign_id:
                continue
            p = F.log_softmax(logits / temperature, dim=1)
            total_loss = total_loss - torch.mean(torch.sum(batch_assign * p, dim=1))
            terms += 1

    loss = total_loss / max(1, terms)

    counts_sum = proto_counts.sum().clamp_min(1.0)
    proto_prob = proto_counts / counts_sum
    used_mask = proto_counts > 0
    used_prob = proto_prob[used_mask]
    entropy = torch.zeros((), device=proto_counts.device)
    if used_prob.numel() > 0:
        entropy = -(used_prob * torch.log(used_prob.clamp_min(1e-12))).sum()
    norm = math.log(proto_counts.numel()) if proto_counts.numel() > 1 else 1.0
    entropy_norm = float((entropy / max(norm, 1e-12)).item())
    active_ratio = float(used_mask.float().mean().item())
    max_usage_ratio = float(proto_prob.max().item())
    stats = {
        "proto_usage_entropy_norm": entropy_norm,
        "proto_active_ratio": active_ratio,
        "proto_max_usage_ratio": max_usage_ratio,
    }
    return loss, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwAV pretraining (multi-crop + queue + optimizer/lr scaling) on STL10.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./outputs/swav_stl10_mcq")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--local-crop-size", type=int, default=48)
    parser.add_argument("--local-crops-number", type=int, default=4)
    parser.add_argument("--global-crops-scale", type=float, nargs=2, default=(0.14, 1.0))
    parser.add_argument("--local-crops-scale", type=float, nargs=2, default=(0.05, 0.14))

    parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    parser.add_argument("--projection-out-dim", type=int, default=128)
    parser.add_argument("--num-prototypes", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--assignment-crop-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--queue-length", type=int, default=3840)
    parser.add_argument("--epoch-queue", type=int, default=50, help="Queue is enabled for epochs <= this value.")
    parser.add_argument("--freeze-prototypes-niters", type=int, default=313)

    parser.add_argument("--optimizer", type=str, choices=["lars", "sgd", "LARS", "SGD"], default="lars")
    parser.add_argument("--base-lr", type=float, default=0.3, help="LR before linear scaling by batch size.")
    parser.add_argument("--final-lr", type=float, default=6e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tb-log-dir", type=str, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--linear-eval-epochs", type=int, default=10)
    parser.add_argument("--linear-eval-lr", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.1)
    parser.add_argument("--periodic-keep-last", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)

    args = parser.parse_args(argv)
    args.assignment_crop_ids = list(parse_assignment_crop_ids(args.assignment_crop_ids))
    args.optimizer = args.optimizer.lower()
    return args


def train_one_epoch(
    model: SwAVModel,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    queue: SwAVQueue | None,
    writer: SummaryWriter | None = None,
) -> tuple[float, int, dict[str, float]]:
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    assignment_crop_ids = parse_assignment_crop_ids(args.assignment_crop_ids)
    use_queue = queue is not None and epoch <= args.epoch_queue
    running_entropy = 0.0
    running_active_ratio = 0.0
    running_max_usage = 0.0

    disable_pbar = args.disable_tqdm or (not sys.stderr.isatty())
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, disable=disable_pbar)

    for step, (views, _targets) in enumerate(pbar):
        crops = [v.to(device, non_blocking=True) for v in views]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=args.amp):
            crop_embeddings, crop_logits = model.forward_crops(crops)
        with amp.autocast("cuda", enabled=False):
            loss, proto_stats = swav_loss_with_queue(
                model=model,
                crop_embeddings=crop_embeddings,
                crop_logits=[x.float() for x in crop_logits],
                assignment_crop_ids=assignment_crop_ids,
                temperature=args.temperature,
                sinkhorn_epsilon=args.sinkhorn_epsilon,
                sinkhorn_iterations=args.sinkhorn_iterations,
                queue=queue,
                use_queue=use_queue,
            )

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        if global_step < args.freeze_prototypes_niters:
            for name, param in model.named_parameters():
                if "prototypes" in name and param.grad is not None:
                    param.grad.zero_()
        scaler.step(optimizer)
        scaler.update()
        if (not args.amp) or (scaler.get_scale() >= old_scale):
            scheduler.step()

        if use_queue and queue is not None:
            with torch.no_grad():
                for cid in assignment_crop_ids:
                    queue.enqueue(cid, crop_embeddings[cid].detach())

        running_loss += loss.item()
        running_entropy += proto_stats["proto_usage_entropy_norm"]
        running_active_ratio += proto_stats["proto_active_ratio"]
        running_max_usage += proto_stats["proto_max_usage_ratio"]
        global_step += 1

        if (step + 1) % args.log_interval == 0 or (step + 1) == num_batches:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / (step + 1)
            if writer is not None:
                writer.add_scalar("train/loss_step", avg_loss, global_step)
                writer.add_scalar("train/lr_step", lr, global_step)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.6f}"})

    denom = max(1, num_batches)
    metrics = {
        "proto_usage_entropy_norm": running_entropy / denom,
        "proto_active_ratio": running_active_ratio / denom,
        "proto_max_usage_ratio": running_max_usage / denom,
    }
    return running_loss / denom, global_step, metrics


def main() -> None:
    args = parse_args()
    device = setup_cuda(args.gpu_id)
    seed_everything(args.seed)

    effective_lr = args.base_lr * (args.batch_size / 256.0)
    args.lr = effective_lr

    train_loader = build_stl10_multicrop_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        local_crop_size=args.local_crop_size,
        local_crops_number=args.local_crops_number,
        global_crops_scale=tuple(args.global_crops_scale),
        local_crops_scale=tuple(args.local_crops_scale),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    train_eval_loader = None
    test_eval_loader = None
    if args.eval_interval > 0:
        train_eval_loader, test_eval_loader = build_stl10_eval_loaders(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )

    model = SwAVModel(
        backbone_name="resnet50",
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
        num_prototypes=args.num_prototypes,
    ).to(device)

    if args.optimizer == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=effective_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=effective_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        final_lr=args.final_lr,
    )
    scaler = amp.GradScaler("cuda", enabled=args.amp)
    queue = None
    if args.queue_length > 0:
        queue = SwAVQueue(
            size=args.queue_length,
            emb_dim=args.projection_out_dim,
            assignment_crop_ids=parse_assignment_crop_ids(args.assignment_crop_ids),
            device=device,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["effective_lr"] = effective_lr
    dump_config(save_dir=str(save_dir), config=config)

    writer: SummaryWriter | None = None
    if args.tensorboard:
        tb_log_dir = Path(args.tb_log_dir) if args.tb_log_dir is not None else (save_dir / "tensorboard")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logging enabled: {tb_log_dir}")

    best_loss = math.inf
    global_step = 0
    periodic_paths: list[Path] = []

    print("=" * 80)
    print("SwAV (multi-crop + queue + optimizer/lr scaling) on STL10 unlabeled")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, effective_lr={effective_lr:.6f}")
    print(
        f"optimizer={args.optimizer}, local_crops={args.local_crops_number}, "
        f"queue_length={args.queue_length}, epoch_queue={args.epoch_queue}, "
        f"freeze_prototypes_niters={args.freeze_prototypes_niters}"
    )
    print("=" * 80)

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss, global_step, proto_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            global_step=global_step,
            queue=queue,
            writer=writer,
        )

        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch:03d}] loss={epoch_loss:.5f} time={elapsed:.1f}s")
        if writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
            writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("train/epoch_time_sec", elapsed, epoch)
            writer.add_scalar("train/proto_usage_entropy_norm", proto_metrics["proto_usage_entropy_norm"], epoch)
            writer.add_scalar("train/proto_active_ratio", proto_metrics["proto_active_ratio"], epoch)
            writer.add_scalar("train/proto_max_usage_ratio", proto_metrics["proto_max_usage_ratio"], epoch)

        state = TrainState(epoch=epoch, best_loss=best_loss, global_step=global_step)
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            state.best_loss = best_loss
            path = save_dir / "backbone_best.pt"
            torch.save(model.backbone.state_dict(), path)
            print(f"  -> best loss updated. backbone saved: {path}")
            if writer is not None:
                writer.add_scalar("train/best_loss", best_loss, epoch)

        save_checkpoint(
            save_dir=str(save_dir),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            is_best=is_best,
        )

        if epoch % args.save_every == 0:
            path = Path(save_backbone_weights(str(save_dir), model, epoch))
            periodic_paths.append(path)
            if args.periodic_keep_last > 0 and len(periodic_paths) > args.periodic_keep_last:
                stale = periodic_paths.pop(0)
                if stale.exists():
                    stale.unlink()
            print(f"  -> periodic backbone save: {path}")

        should_eval = args.eval_interval > 0 and train_eval_loader is not None and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        if should_eval:
            eval_start = time.time()
            knn_top1, linear_top1 = run_representation_eval(
                model=model,
                train_eval_loader=train_eval_loader,
                test_eval_loader=test_eval_loader,
                device=device,
                args=args,
            )
            eval_elapsed = time.time() - eval_start
            print(f"  -> eval @ epoch {epoch:03d}: knn_top1={knn_top1:.2f}% linear_top1={linear_top1:.2f}% ({eval_elapsed:.1f}s)")
            if writer is not None:
                writer.add_scalar("eval/knn_top1", knn_top1, epoch)
                writer.add_scalar("eval/linear_top1", linear_top1, epoch)
                writer.add_scalar("eval/time_sec", eval_elapsed, epoch)

    total = time.time() - train_start
    print(f"Training finished in {total / 3600:.2f}h. Best loss={best_loss:.5f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
