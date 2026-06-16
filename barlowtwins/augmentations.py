from __future__ import annotations

from dataclasses import dataclass

import torchvision.transforms as T


class AsymmetricTwoCropsTransform:
    def __init__(self, transform_view1: T.Compose, transform_view2: T.Compose) -> None:
        self.transform_view1 = transform_view1
        self.transform_view2 = transform_view2

    def __call__(self, x):
        return self.transform_view1(x), self.transform_view2(x)


@dataclass
class BarlowTwinsAugmentation:
    image_size: int = 96
    color_jitter_p: float = 0.8
    random_grayscale_p: float = 0.2
    hflip_p: float = 0.5

    # B(alanced) setup with paper-style asymmetric views.
    blur_p_view1: float = 1.0
    blur_p_view2: float = 0.1
    solarization_p_view1: float = 0.0
    solarization_p_view2: float = 0.2

    def _common_ops(self) -> list:
        color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        )
        return [
            T.RandomResizedCrop(
                self.image_size,
                scale=(0.2, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=self.hflip_p),
            T.RandomApply([color_jitter], p=self.color_jitter_p),
            T.RandomGrayscale(p=self.random_grayscale_p),
        ]

    def _blur_op(self) -> T.GaussianBlur:
        blur_kernel = max(3, int(self.image_size * 0.1))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        return T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))

    def _post_ops(self) -> list:
        return [
            T.ToTensor(),
            T.Normalize(
                mean=(0.4467, 0.4398, 0.4066),
                std=(0.2603, 0.2566, 0.2713),
            ),
        ]

    def build_view1(self) -> T.Compose:
        return T.Compose(
            self._common_ops()
            + [
                T.RandomApply([self._blur_op()], p=self.blur_p_view1),
                T.RandomSolarize(threshold=128, p=self.solarization_p_view1),
            ]
            + self._post_ops()
        )

    def build_view2(self) -> T.Compose:
        return T.Compose(
            self._common_ops()
            + [
                T.RandomApply([self._blur_op()], p=self.blur_p_view2),
                T.RandomSolarize(threshold=128, p=self.solarization_p_view2),
            ]
            + self._post_ops()
        )
