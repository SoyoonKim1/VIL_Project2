from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    SimCLR NT-Xent loss for single GPU training.
    Input z1/z2 must be L2-normalized.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 shapes must match. got {z1.shape} vs {z2.shape}")
        batch_size = z1.size(0)
        if batch_size < 2:
            raise ValueError("Batch size must be >= 2 for contrastive learning.")

        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.matmul(z, z.T) / self.temperature

        # mask self-similarity
        logits_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        # Keep mask value representable across dtypes (e.g., fp16 under AMP).
        mask_value = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(~logits_mask, mask_value)

        # positive pair indices
        positives = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ],
            dim=0,
        )

        loss = F.cross_entropy(sim, positives)
        return loss
