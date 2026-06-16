from __future__ import annotations

import unittest

import torch

from swav import SwAVLoss, SwAVModel, sinkhorn_knopp


class TestSwAV(unittest.TestCase):
    def test_sinkhorn_output_is_row_normalized(self) -> None:
        scores = torch.randn(8, 16)
        q = sinkhorn_knopp(scores, epsilon=0.05, iterations=3)
        self.assertEqual(q.shape, (8, 16))
        row_sums = q.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4))

    def test_swav_loss_returns_scalar(self) -> None:
        logits = [torch.randn(8, 32), torch.randn(8, 32), torch.randn(8, 32)]
        criterion = SwAVLoss(temperature=0.1, sinkhorn_epsilon=0.05, sinkhorn_iterations=3)
        loss = criterion(logits)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_swav_model_forward_shapes(self) -> None:
        model = SwAVModel(
            backbone_name="resnet50",
            projection_hidden_dim=64,
            projection_out_dim=32,
            num_prototypes=50,
        )
        x = torch.randn(2, 3, 96, 96)
        z, logits = model(x)
        self.assertEqual(z.shape, (2, 32))
        self.assertEqual(logits.shape, (2, 50))


if __name__ == "__main__":
    unittest.main()
