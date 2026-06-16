from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import STL10

from .augmentations import AsymmetricTwoCropsTransform, BarlowTwinsAugmentation


def build_stl10_barlowtwins_unlabeled_loader(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    blur_p_view1: float = 1.0,
    blur_p_view2: float = 0.1,
    solarization_p_view1: float = 0.0,
    solarization_p_view2: float = 0.2,
    pin_memory: bool = True,
    drop_last: bool = True,
    download: bool = False,
):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    aug = BarlowTwinsAugmentation(
        image_size=image_size,
        blur_p_view1=blur_p_view1,
        blur_p_view2=blur_p_view2,
        solarization_p_view1=solarization_p_view1,
        solarization_p_view2=solarization_p_view2,
    )
    transform = AsymmetricTwoCropsTransform(
        transform_view1=aug.build_view1(),
        transform_view2=aug.build_view2(),
    )
    dataset = STL10(
        root=str(root),
        split="unlabeled",
        transform=transform,
        download=download,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
