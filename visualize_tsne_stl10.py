from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from simclr.datasets import build_stl10_linear_eval_loaders
from simclr.model import SimCLRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SimCLR features on STL10 with t-SNE.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory that contains train_config.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint/backbone weights. If omitted, searched in run-dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "test", "both"),
        default="test",
        help="Which split to visualize.",
    )
    parser.add_argument("--max-samples", type=int, default=3000, help="Max samples per split for t-SNE.")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output png path. Default: <run-dir>/tsne_<split>.png",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for feature extraction.",
    )
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        ckpt = Path(checkpoint_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    candidates = [
        run_dir / "backbone_best.pt",
        run_dir / "checkpoint_best.pt",
        run_dir / "checkpoint_last.pt",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find checkpoint automatically. "
        "Pass --checkpoint explicitly.\nSearched:\n"
        f"{searched}"
    )


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: dict, device: torch.device) -> SimCLRModel:
    model = SimCLRModel(
        backbone_name="resnet50",
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        projection_out_dim=int(config["projection_out_dim"]),
    ).to(device)
    model.eval()
    return model


def load_backbone_weights(model: SimCLRModel, checkpoint_path: Path, device: torch.device) -> None:
    obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(obj, dict) and "model" in obj:
        model_state = obj["model"]
        model.load_state_dict(model_state, strict=False)
        return

    if isinstance(obj, dict):
        if any(k.startswith("layer") or k.startswith("conv1") for k in obj.keys()):
            model.backbone.load_state_dict(obj, strict=True)
            return

    raise ValueError(
        f"Unsupported checkpoint format in {checkpoint_path}. "
        "Expected backbone state_dict or full checkpoint with 'model' key."
    )


@torch.no_grad()
def extract_features(
    model: SimCLRModel,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total = 0
    for images, target in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        out = model.forward_backbone(images).detach().cpu().numpy()
        y = target.numpy()
        feats.append(out)
        labels.append(y)
        total += out.shape[0]
        if total >= max_samples:
            break

    x = np.concatenate(feats, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]
    return x, y


def run_tsne(x: np.ndarray, seed: int, perplexity: float, n_iter: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        init="pca",
        random_state=seed,
    )
    return tsne.fit_transform(x)


def draw_and_save(emb: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    classes = sorted(np.unique(labels).tolist())
    for c in classes:
        idx = labels == c
        plt.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.75, color=cmap(c), label=str(c))
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Class", markerscale=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device(args.device)

    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
    model = build_model(config, device=device)
    load_backbone_weights(model, ckpt_path, device=device)

    train_loader, test_loader = build_stl10_linear_eval_loaders(
        data_root=str(config["data_root"]),
        image_size=int(config["image_size"]),
        batch_size=int(config.get("eval_batch_size", 256)),
        num_workers=int(config["num_workers"]),
        download=bool(config.get("download", False)),
    )

    split_to_loader = {"train": train_loader, "test": test_loader}
    if args.split == "both":
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for split_name in ("train", "test"):
            x, y = extract_features(
                model=model,
                loader=split_to_loader[split_name],
                device=device,
                max_samples=args.max_samples,
            )
            all_x.append(x)
            all_y.append(y)
        features = np.concatenate(all_x, axis=0)
        labels = np.concatenate(all_y, axis=0)
        title = f"STL10 t-SNE ({run_dir.name}, split=both)"
    else:
        features, labels = extract_features(
            model=model,
            loader=split_to_loader[args.split],
            device=device,
            max_samples=args.max_samples,
        )
        title = f"STL10 t-SNE ({run_dir.name}, split={args.split})"

    emb = run_tsne(
        x=features,
        seed=args.seed,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )

    out_path = Path(args.out) if args.out is not None else (run_dir / f"tsne_{args.split}.png")
    draw_and_save(emb=emb, labels=labels, out_path=out_path, title=title)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == "__main__":
    main()
