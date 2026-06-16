from __future__ import annotations

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


class SimCLRModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_hidden_dim: int = 2048,
        projection_out_dim: int = 128,
    ) -> None:
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError(f"Only resnet50 is supported in this template, got {backbone_name}.")

        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features  # 2048 for ResNet-50
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_backbone(x)
        projections = self.projector(features)
        projections = F.normalize(projections, dim=1)
        return features, projections
