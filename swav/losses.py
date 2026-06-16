from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, epsilon: float = 0.05, iterations: int = 3) -> torch.Tensor:
    """
    Compute balanced assignments from prototype scores.

    Args:
        scores: [batch_size, num_prototypes]
    Returns:
        assignments: [batch_size, num_prototypes] where each row sums to 1.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D [B, K], got shape {tuple(scores.shape)}")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")

    # Sinkhorn is numerically sensitive under AMP/fp16.
    # Compute in fp32 with max-shift stabilization before exp.
    scores_fp32 = scores.float()
    scores_fp32 = scores_fp32 - scores_fp32.max(dim=1, keepdim=True).values

    q = torch.exp(scores_fp32 / epsilon).T  # [K, B]
    q = q / q.sum().clamp_min(1e-12)

    k, b = q.shape
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q / k
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q / b

    q = q * b
    return q.T.contiguous()


class SwAVLoss(nn.Module):
    """
    SwAV multi-crop loss with online Sinkhorn assignments.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        assignment_crop_ids: tuple[int, ...] = (0, 1),
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        if len(assignment_crop_ids) == 0:
            raise ValueError("assignment_crop_ids must not be empty.")
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.assignment_crop_ids = assignment_crop_ids

    def forward(self, crop_logits: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(crop_logits) < 2:
            raise ValueError("SwAV loss expects at least 2 crop logits.")

        for idx, logits in enumerate(crop_logits):
            if logits.ndim != 2:
                raise ValueError(f"crop_logits[{idx}] must be 2D [B, K], got {tuple(logits.shape)}")

        num_crops = len(crop_logits)
        total_loss = torch.zeros((), device=crop_logits[0].device)
        terms = 0

        for assign_id in self.assignment_crop_ids:
            if assign_id < 0 or assign_id >= num_crops:
                raise ValueError(f"assignment crop id {assign_id} out of range for {num_crops} crops.")

            with torch.no_grad():
                q = sinkhorn_knopp(
                    crop_logits[assign_id].detach(),
                    epsilon=self.sinkhorn_epsilon,
                    iterations=self.sinkhorn_iterations,
                )

            for pred_id, logits in enumerate(crop_logits):
                if pred_id == assign_id:
                    continue
                p = F.log_softmax(logits / self.temperature, dim=1)
                total_loss = total_loss - torch.mean(torch.sum(q * p, dim=1))
                terms += 1

        return total_loss / max(terms, 1)
