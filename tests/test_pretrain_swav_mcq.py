from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from pretrain_swav_stl10_mcq import MultiCropTransform, parse_args


class TestPretrainSwAVMCQ(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.save_dir, "./outputs/swav_stl10_mcq")
        self.assertEqual(args.optimizer, "lars")
        self.assertEqual(args.local_crops_number, 4)
        self.assertEqual(args.local_crop_size, 48)
        self.assertEqual(args.num_prototypes, 512)
        self.assertEqual(args.queue_length, 3840)
        self.assertEqual(args.epoch_queue, 50)
        self.assertEqual(args.freeze_prototypes_niters, 313)
        self.assertEqual(args.base_lr, 0.3)

    def test_multicrop_transform_returns_expected_count(self) -> None:
        transform = MultiCropTransform(
            image_size=96,
            local_crops_number=4,
            local_crop_size=64,
        )
        img = Image.fromarray((np.random.rand(96, 96, 3) * 255).astype(np.uint8))
        crops = transform(img)
        self.assertEqual(len(crops), 6)  # 2 global + 4 local


if __name__ == "__main__":
    unittest.main()
