from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import STL10

from .augmentations import SimCLRAugmentation, TwoCropsTransform


def build_stl10_unlabeled_loader(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = True,
    download: bool = False,
    color_jitter_strength: float = 0.5,
    gaussian_blur_prob: float = 0.5,
):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    base_transform = SimCLRAugmentation(
        image_size=image_size,
        color_jitter_strength=color_jitter_strength,
        gaussian_blur_prob=gaussian_blur_prob,
    ).build()
    two_crops_transform = TwoCropsTransform(base_transform)

    dataset = STL10(
        root=str(root),
        split="unlabeled",
        transform=two_crops_transform,
        download=download,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    return loader


def build_stl10_linear_eval_loaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    download: bool = False,
):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

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

    train_dataset = STL10(
        root=str(root),
        split="train",
        transform=eval_transform,
        download=download,
    )
    test_dataset = STL10(
        root=str(root),
        split="test",
        transform=eval_transform,
        download=download,
    )

    common_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=False, **common_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_args)
    return train_loader, test_loader
