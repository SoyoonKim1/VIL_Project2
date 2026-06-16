from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class BarlowProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 4096,
        out_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_hidden_dim: int = 4096,
        projection_out_dim: int = 4096,
    ) -> None:
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError(f"Only resnet50 is supported in this project, got {backbone_name}.")

        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projector = BarlowProjector(
            in_dim=feat_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_backbone(x)
        projections = self.projector(features)
        return features, projections
