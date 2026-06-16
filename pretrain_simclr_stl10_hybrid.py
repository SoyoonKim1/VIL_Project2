from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import amp
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pretrain_simclr_stl10 import build_scheduler, run_representation_eval
from simclr.datasets import build_stl10_linear_eval_loaders, build_stl10_unlabeled_loader
from simclr.losses import NTXentLoss
from simclr.model import SimCLRModel
from simclr.utils import TrainState, dump_config, save_backbone_weights, save_checkpoint, seed_everything, setup_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid SimCLR pretraining (contrastive + EMA distill + light masking).")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./outputs/simclr_stl10_hybrid")
    parser.add_argument("--download", action="store_true", help="Download STL10 if missing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="Single GPU id.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    parser.add_argument("--projection-out-dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=0.3, help="Base lr for batch_size=256.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--tb-log-dir", type=str, default=None, help="TensorBoard log dir. Defaults to <save-dir>/tensorboard.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars (recommended for nohup logs).")

    parser.add_argument("--eval-interval", type=int, default=25, help="Run kNN/linear eval every N epochs (0 to disable).")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--linear-eval-epochs", type=int, default=10)
    parser.add_argument("--linear-eval-lr", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-temperature", type=float, default=0.1)
    parser.add_argument("--periodic-keep-last", type=int, default=3, help="Number of periodic backbone checkpoints to keep.")

    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=25)

    # Hybrid options
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Fraction of image patches to zero out per view.")
    parser.add_argument("--mask-patch-size", type=int, default=16, help="Patch size for block masking.")
    parser.add_argument("--distill-weight", type=float, default=1.0, help="Weight for EMA teacher distillation loss.")
    parser.add_argument("--ema-momentum", type=float, default=0.996, help="EMA momentum for teacher update.")

    return parser.parse_args()


def apply_light_block_mask(x: torch.Tensor, mask_ratio: float, patch_size: int) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return x
    if mask_ratio >= 1.0:
        return torch.zeros_like(x)
    if patch_size <= 0:
        raise ValueError("mask-patch-size must be positive.")

    b, c, h, w = x.shape
    gh = math.ceil(h / patch_size)
    gw = math.ceil(w / patch_size)
    pad_h = gh * patch_size - h
    pad_w = gw * patch_size - w
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    patches = x_pad.view(b, c, gh, patch_size, gw, patch_size).permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(b, gh * gw, c, patch_size, patch_size)

    mask = torch.rand(b, gh * gw, device=x.device) < mask_ratio
    patches[mask] = 0.0

    x_masked = patches.view(b, gh, gw, c, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).contiguous()
    x_masked = x_masked.view(b, c, gh * patch_size, gw * patch_size)
    return x_masked[:, :, :h, :w]


@torch.no_grad()
def update_ema_teacher(student: SimCLRModel, teacher: SimCLRModel, ema_momentum: float) -> None:
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.mul_(ema_momentum).add_(s_param.detach(), alpha=1.0 - ema_momentum)


def distill_loss(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    s = F.normalize(student_z, dim=1)
    t = F.normalize(teacher_z, dim=1)
    return F.mse_loss(s, t)


def train_one_epoch_hybrid(
    student: SimCLRModel,
    teacher: SimCLRModel,
    loader: torch.utils.data.DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    writer: SummaryWriter | None = None,
) -> tuple[float, int]:
    student.train()
    teacher.eval()

    running_total = 0.0
    running_contrastive = 0.0
    running_distill = 0.0
    num_batches = len(loader)

    disable_pbar = args.disable_tqdm or (not sys.stderr.isatty())
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, disable=disable_pbar)
    for step, (views, _targets) in enumerate(pbar):
        x1, x2 = views
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        x1_masked = apply_light_block_mask(x1, mask_ratio=args.mask_ratio, patch_size=args.mask_patch_size)
        x2_masked = apply_light_block_mask(x2, mask_ratio=args.mask_ratio, patch_size=args.mask_patch_size)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=args.amp):
            _, z1_student = student(x1_masked)
            _, z2_student = student(x2_masked)
            contrastive = criterion(z1_student, z2_student)
            with torch.no_grad():
                _, z1_teacher = teacher(x1)
                _, z2_teacher = teacher(x2)
            distill = 0.5 * (distill_loss(z1_student, z1_teacher) + distill_loss(z2_student, z2_teacher))
            total_loss = contrastive + args.distill_weight * distill

        old_scale = scaler.get_scale()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step scheduler and EMA only when optimizer step is effectively applied.
        if (not args.amp) or (scaler.get_scale() >= old_scale):
            scheduler.step()
            update_ema_teacher(student, teacher, ema_momentum=args.ema_momentum)

        running_total += total_loss.item()
        running_contrastive += contrastive.item()
        running_distill += distill.item()
        global_step += 1

        if (step + 1) % args.log_interval == 0 or (step + 1) == num_batches:
            lr = optimizer.param_groups[0]["lr"]
            avg_total = running_total / (step + 1)
            avg_contrastive = running_contrastive / (step + 1)
            avg_distill = running_distill / (step + 1)
            if writer is not None:
                writer.add_scalar("train/loss_step", avg_total, global_step)
                writer.add_scalar("train/loss_contrastive_step", avg_contrastive, global_step)
                writer.add_scalar("train/loss_distill_step", avg_distill, global_step)
                writer.add_scalar("train/lr_step", lr, global_step)
            pbar.set_postfix(
                {
                    "loss": f"{avg_total:.4f}",
                    "ctr": f"{avg_contrastive:.4f}",
                    "dst": f"{avg_distill:.4f}",
                    "lr": f"{lr:.6f}",
                }
            )

    return running_total / max(1, num_batches), global_step


def main() -> None:
    args = parse_args()
    device = setup_cuda(args.gpu_id)
    seed_everything(args.seed)

    eff_lr = args.lr * (args.batch_size / 256.0)

    loader = build_stl10_unlabeled_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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

    student = SimCLRModel(
        backbone_name="resnet50",
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
    ).to(device)
    teacher = SimCLRModel(
        backbone_name="resnet50",
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    criterion = NTXentLoss(temperature=args.temperature)

    optimizer = SGD(
        student.parameters(),
        lr=eff_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(loader), args=args)
    scaler = amp.GradScaler("cuda", enabled=args.amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dump_config(save_dir=str(save_dir), config=vars(args))

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
    print("Hybrid SimCLR pretraining on STL10 unlabeled")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={eff_lr:.6f}")
    print(
        f"mask_ratio={args.mask_ratio}, mask_patch_size={args.mask_patch_size}, "
        f"distill_weight={args.distill_weight}, ema_momentum={args.ema_momentum}"
    )
    print("=" * 80)

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_loss, global_step = train_one_epoch_hybrid(
            student=student,
            teacher=teacher,
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
            torch.save(student.backbone.state_dict(), path)
            print(f"  -> best loss updated. backbone saved: {path}")
            if writer is not None:
                writer.add_scalar("train/best_loss", best_loss, epoch)

        save_checkpoint(
            save_dir=str(save_dir),
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state,
            is_best=is_best,
        )

        if epoch % args.save_every == 0:
            path = Path(save_backbone_weights(str(save_dir), student, epoch))
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
                model=student,
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
