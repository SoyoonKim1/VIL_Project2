from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 2048,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwAVModel(nn.Module):
    """
    SwAV-style backbone + projection + prototype heads.

    - backbone: feature extractor (ResNet-50 by default)
    - projector: maps backbone features to embedding space
    - prototypes: maps normalized embeddings to prototype logits
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_hidden_dim: int = 2048,
        projection_out_dim: int = 128,
        num_prototypes: int = 3000,
        normalize_prototypes: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError(f"Only resnet50 is supported in this template, got {backbone_name}.")
        if num_prototypes <= 0:
            raise ValueError(f"num_prototypes must be positive, got {num_prototypes}.")

        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )
        self.prototypes = nn.Linear(projection_out_dim, num_prototypes, bias=False)
        self.normalize_prototypes = normalize_prototypes

    @torch.no_grad()
    def _normalize_prototypes_if_needed(self) -> None:
        if self.normalize_prototypes:
            w = self.prototypes.weight.data
            self.prototypes.weight.copy_(F.normalize(w, dim=1))

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_backbone(x)
        z = self.projector(features)
        z = F.normalize(z, dim=1)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_embeddings(x)
        self._normalize_prototypes_if_needed()
        logits = self.prototypes(z)
        return z, logits

    def forward_crops(self, crops: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        embeddings: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for crop in crops:
            z, p = self.forward(crop)
            embeddings.append(z)
            logits.append(p)
        return embeddings, logits
