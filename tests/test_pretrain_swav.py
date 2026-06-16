from __future__ import annotations

import unittest

from pretrain_swav_stl10 import parse_args, parse_assignment_crop_ids


class TestPretrainSwAVScript(unittest.TestCase):
    def test_parse_assignment_crop_ids(self) -> None:
        ids = parse_assignment_crop_ids([0, 2, 3])
        self.assertEqual(ids, (0, 2, 3))

    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.save_dir, "./outputs/swav_stl10")
        self.assertEqual(args.num_prototypes, 3000)
        self.assertEqual(args.assignment_crop_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
