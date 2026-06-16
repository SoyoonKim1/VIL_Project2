from __future__ import annotations

from dataclasses import dataclass

import torchvision.transforms as T


@dataclass
class SimCLRAugmentation:
    image_size: int = 96
    color_jitter_strength: float = 0.5
    gaussian_blur_prob: float = 0.5
    solarization_prob: float = 0.0

    def build(self) -> T.Compose:
        cj = self.color_jitter_strength
        color_jitter = T.ColorJitter(
            brightness=0.8 * cj,
            contrast=0.8 * cj,
            saturation=0.8 * cj,
            hue=0.2 * cj,
        )

        blur_kernel = max(3, int(self.image_size * 0.1))
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        return T.Compose(
            [
                T.RandomResizedCrop(
                    self.image_size,
                    scale=(0.2, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))],
                    p=self.gaussian_blur_prob,
                ),
                T.RandomSolarize(threshold=128, p=self.solarization_prob),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4467, 0.4398, 0.4066),
                    std=(0.2603, 0.2566, 0.2713),
                ),
            ]
        )


class TwoCropsTransform:
    def __init__(self, base_transform: T.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
