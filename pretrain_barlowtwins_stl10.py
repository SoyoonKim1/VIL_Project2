from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import amp
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from barlowtwins.datasets import build_stl10_barlowtwins_unlabeled_loader
from barlowtwins.losses import BarlowTwinsLoss
from barlowtwins.model import BarlowTwinsModel
from simclr.datasets import build_stl10_linear_eval_loaders
from simclr.utils import TrainState, dump_config, save_backbone_weights, save_checkpoint, seed_everything, setup_cuda


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
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, eps=eps)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barlow Twins (balanced profile) pretraining on STL10 unlabeled split.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./outputs/barlowtwins_stl10_b_gpu2_seed42")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=2, help="Single GPU id.")

    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--projection-hidden-dim", type=int, default=4096)
    parser.add_argument("--projection-out-dim", type=int, default=4096)
    parser.add_argument("--lambda-offdiag", type=float, default=5e-3)

    # Keep B profile fixed by default; still overridable for ablations.
    parser.add_argument("--blur-p-view1", type=float, default=1.0)
    parser.add_argument("--blur-p-view2", type=float, default=0.1)
    parser.add_argument("--solarization-p-view1", type=float, default=0.0)
    parser.add_argument("--solarization-p-view2", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, choices=["lars", "sgd"], default="lars")
    parser.add_argument("--lr", type=float, default=0.6, help="Optimizer learning rate.")
    parser.add_argument("--final-lr", type=float, default=6e-4, help="Cosine scheduler eta_min.")
    parser.add_argument(
        "--scale-lr-by-batch",
        action="store_true",
        help="Scale lr/final-lr by effective_batch_size/256. Disabled by default for stable single-GPU runs.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tb-log-dir", type=str, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--linear-eval-epochs", type=int, default=10)
    parser.add_argument("--linear-eval-lr", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.1)
    parser.add_argument("--periodic-keep-last", type=int, default=3)

    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, eta_min: float):
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train_one_epoch(
    model: BarlowTwinsModel,
    loader: torch.utils.data.DataLoader,
    criterion: BarlowTwinsLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    writer: SummaryWriter | None = None,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0
    num_batches = len(loader)
    accum_steps = max(1, args.grad_accum_steps)

    disable_pbar = args.disable_tqdm or (not sys.stderr.isatty())
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, disable=disable_pbar)
    optimizer.zero_grad(set_to_none=True)

    for step, (views, _targets) in enumerate(pbar):
        x1, x2 = views
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=args.amp):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            loss_for_backward = loss / accum_steps

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at epoch={epoch}, step={step + 1}. "
                "Try a lower lr or disable lr batch scaling."
            )

        old_scale = scaler.get_scale()
        scaler.scale(loss_for_backward).backward()

        should_step = ((step + 1) % accum_steps == 0) or ((step + 1) == num_batches)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if (not args.amp) or (scaler.get_scale() >= old_scale):
                scheduler.step()

        running_loss += loss.item()
        global_step += 1

        if (step + 1) % args.log_interval == 0 or (step + 1) == num_batches:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / (step + 1)
            if writer is not None:
                writer.add_scalar("train/loss_step", avg_loss, global_step)
                writer.add_scalar("train/lr_step", lr, global_step)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.6f}"})

    return running_loss / max(1, num_batches), global_step


@torch.no_grad()
def extract_backbone_features(
    model: BarlowTwinsModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        features = model.forward_backbone(images)
        feats.append(features.detach().cpu())
        labels.append(target.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def knn_eval(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = 10,
    k: int = 20,
    temperature: float = 0.1,
    chunk_size: int = 512,
) -> float:
    train_feats = F.normalize(train_feats, dim=1)
    test_feats = F.normalize(test_feats, dim=1)

    total = test_feats.size(0)
    correct = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        query = test_feats[start:end]
        sim = torch.matmul(query, train_feats.T)
        topk_sim, topk_idx = sim.topk(k=min(k, train_feats.size(0)), dim=1)
        topk_labels = train_labels[topk_idx]

        weights = torch.exp(topk_sim / temperature)
        votes = torch.zeros(query.size(0), num_classes, dtype=weights.dtype)
        votes.scatter_add_(1, topk_labels, weights)
        pred = votes.argmax(dim=1)
        correct += (pred == test_labels[start:end]).sum().item()
    return 100.0 * correct / max(1, total)


def linear_eval(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.1,
    weight_decay: float = 0.0,
) -> float:
    in_dim = train_feats.size(1)
    num_classes = int(train_labels.max().item() + 1)
    classifier = torch.nn.Linear(in_dim, num_classes).to(device)
    optimizer = SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    train_ds = TensorDataset(train_feats, train_labels)
    test_ds = TensorDataset(test_feats, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    for _ in range(max(1, epochs)):
        classifier.train()
        for feats, target in train_loader:
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, target in test_loader:
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = classifier(feats).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    return 100.0 * correct / max(1, total)


def run_representation_eval(
    model: BarlowTwinsModel,
    train_eval_loader: torch.utils.data.DataLoader,
    test_eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float]:
    train_feats, train_labels = extract_backbone_features(model, train_eval_loader, device)
    test_feats, test_labels = extract_backbone_features(model, test_eval_loader, device)

    knn_top1 = knn_eval(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        k=args.knn_k,
        temperature=args.knn_temperature,
    )
    linear_top1 = linear_eval(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        device=device,
        epochs=args.linear_eval_epochs,
        batch_size=args.eval_batch_size,
        lr=args.linear_eval_lr,
    )
    return knn_top1, linear_top1


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    device = setup_cuda(args.gpu_id)
    seed_everything(args.seed)

    loader = build_stl10_barlowtwins_unlabeled_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        blur_p_view1=args.blur_p_view1,
        blur_p_view2=args.blur_p_view2,
        solarization_p_view1=args.solarization_p_view1,
        solarization_p_view2=args.solarization_p_view2,
        download=args.download,
    )
    train_eval_loader = None
    test_eval_loader = None
    if args.eval_interval > 0:
        train_eval_loader, test_eval_loader = build_stl10_linear_eval_loaders(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            download=args.download,
        )

    effective_batch_size = args.batch_size * args.grad_accum_steps
    if args.scale_lr_by_batch:
        lr_scale = effective_batch_size / 256.0
    else:
        lr_scale = 1.0
    eff_lr = args.lr * lr_scale
    eff_final_lr = args.final_lr * lr_scale

    model = BarlowTwinsModel(
        backbone_name="resnet50",
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
    ).to(device)
    criterion = BarlowTwinsLoss(lambda_offdiag=args.lambda_offdiag)

    if args.optimizer == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=eff_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=eff_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_optimizer_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        eta_min=eff_final_lr,
    )
    scaler = amp.GradScaler("cuda", enabled=args.amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dump_config(save_dir=str(save_dir), config={**vars(args), "effective_batch_size": effective_batch_size, "effective_lr": eff_lr})

    writer: SummaryWriter | None = None
    if args.tensorboard:
        tb_log_dir = Path(args.tb_log_dir) if args.tb_log_dir is not None else (save_dir / "tensorboard")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logging enabled: {tb_log_dir}")

    best_loss = math.inf
    global_step = 0
    periodic_paths: list[Path] = []

    print("=" * 100)
    print("Barlow Twins pretraining on STL10 unlabeled (balanced single-GPU profile)")
    print(
        f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"grad_accum_steps={args.grad_accum_steps}, effective_batch={effective_batch_size}, "
        f"optimizer={args.optimizer}, lr={eff_lr:.6f}, final_lr={eff_final_lr:.6f}, "
        f"scale_lr_by_batch={args.scale_lr_by_batch}"
    )
    print(
        "asymmetric augmentation: "
        f"blur(view1/view2)=({args.blur_p_view1:.2f}/{args.blur_p_view2:.2f}), "
        f"solarization(view1/view2)=({args.solarization_p_view1:.2f}/{args.solarization_p_view2:.2f})"
    )
    print("=" * 100)

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss, global_step = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            global_step=global_step,
            writer=writer,
        )

        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch:03d}] loss={epoch_loss:.5f} time={elapsed:.1f}s")
        if writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
            writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("train/epoch_time_sec", elapsed, epoch)

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

        should_eval = args.eval_interval > 0 and train_eval_loader is not None and (
            epoch % args.eval_interval == 0 or epoch == args.epochs
        )
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
