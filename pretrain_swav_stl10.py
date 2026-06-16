from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import amp
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pretrain_simclr_stl10 import build_scheduler, run_representation_eval
from simclr.datasets import build_stl10_linear_eval_loaders, build_stl10_unlabeled_loader
from simclr.utils import TrainState, dump_config, save_backbone_weights, save_checkpoint, seed_everything, setup_cuda
from swav import SwAVLoss, SwAVModel


def parse_assignment_crop_ids(crop_ids: Sequence[int]) -> tuple[int, ...]:
    if len(crop_ids) == 0:
        raise ValueError("assignment-crop-ids must not be empty.")
    if len(set(crop_ids)) != len(crop_ids):
        raise ValueError(f"assignment-crop-ids must be unique, got {list(crop_ids)}")
    if any(i < 0 for i in crop_ids):
        raise ValueError(f"assignment-crop-ids must be non-negative, got {list(crop_ids)}")
    return tuple(crop_ids)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwAV pretraining on STL10 unlabeled split.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./outputs/swav_stl10")
    parser.add_argument("--download", action="store_true", help="Download STL10 if missing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="Single GPU id.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=96)

    parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    parser.add_argument("--projection-out-dim", type=int, default=128)
    parser.add_argument("--num-prototypes", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument(
        "--assignment-crop-ids",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Crop indices used to build Sinkhorn assignments.",
    )

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
    args = parser.parse_args(argv)
    args.assignment_crop_ids = list(parse_assignment_crop_ids(args.assignment_crop_ids))
    return args


def train_one_epoch(
    model: SwAVModel,
    loader: torch.utils.data.DataLoader,
    criterion: SwAVLoss,
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

    disable_pbar = args.disable_tqdm or (not sys.stderr.isatty())
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False, disable=disable_pbar)
    for step, (views, _targets) in enumerate(pbar):
        crops = [v.to(device, non_blocking=True) for v in views]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=args.amp):
            _, crop_logits = model.forward_crops(crops)
            loss = criterion(crop_logits)

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.6f}",
                }
            )

    return running_loss / max(1, num_batches), global_step


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

    model = SwAVModel(
        backbone_name="resnet50",
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
        num_prototypes=args.num_prototypes,
    ).to(device)
    criterion = SwAVLoss(
        temperature=args.temperature,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        assignment_crop_ids=parse_assignment_crop_ids(args.assignment_crop_ids),
    )

    optimizer = SGD(
        model.parameters(),
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
    print("SwAV pretraining on STL10 unlabeled")
    print(f"device={device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={eff_lr:.6f}")
    print(
        "num_prototypes="
        f"{args.num_prototypes}, sinkhorn_eps={args.sinkhorn_epsilon}, "
        f"sinkhorn_iters={args.sinkhorn_iterations}, assignment_crop_ids={args.assignment_crop_ids}"
    )
    print("=" * 80)

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
