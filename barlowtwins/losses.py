from __future__ import annotations

import torch
import torch.nn as nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError(f"off_diagonal expects square matrix, got {n}x{m}.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_offdiag: float = 5e-3, eps: float = 1e-12) -> None:
        super().__init__()
        self.lambda_offdiag = lambda_offdiag
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: z1={tuple(z1.shape)} z2={tuple(z2.shape)}")

        batch_size = z1.size(0)
        if batch_size < 2:
            raise ValueError("Barlow Twins requires batch size >= 2.")

        z1 = (z1 - z1.mean(dim=0)) / torch.sqrt(z1.var(dim=0, unbiased=False) + self.eps)
        z2 = (z2 - z2.mean(dim=0)) / torch.sqrt(z2.var(dim=0, unbiased=False) + self.eps)

        c = (z1.T @ z2) / batch_size

        on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
        off_diag_loss = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_offdiag * off_diag_loss
