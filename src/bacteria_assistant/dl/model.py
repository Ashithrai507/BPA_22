from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_pred = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_pred.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def _pooled_feature_dim(backbone_name: str) -> int:
    dims = {
        "efficientnet_b0": 1280,
        "resnet18": 512,
        "resnet50": 2048,
        "mobilenet_v3_large": 960,
    }
    return dims[backbone_name]


class EmbeddingModel(nn.Module):
    """Pretrained backbone + multi-task heads producing a learned image embedding.

    The globally-pooled embedding replaces the hand-engineered image feature
    vector; the heads are used only during training (as regularizers + for the
    species task). Downstream sklearn classifiers consume the embedding.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_species: int = 10,
        num_groups: int = 4,
        num_grams: int = 3,
        num_types: int = 2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = _pooled_feature_dim(backbone_name)

        backbone = self._build_backbone(backbone_name, pretrained)
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.species_head = nn.Linear(self.embedding_dim, num_species)
        self.group_head = nn.Linear(self.embedding_dim, num_groups)
        self.gram_head = nn.Linear(self.embedding_dim, num_grams)
        self.type_head = nn.Linear(self.embedding_dim, num_types)

    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone_name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            features = net.features
        elif backbone_name == "resnet18":
            net = models.resnet18(weights=weights)
            features = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4,
            )
        elif backbone_name == "mobilenet_v3_large":
            net = models.mobilenet_v3_large(weights=weights)
            features = net.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        return features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.pool(self.backbone(x))
        emb = torch.flatten(features, 1)
        logits = {
            "species": self.species_head(emb),
            "group": self.group_head(emb),
            "gram": self.gram_head(emb),
            "type": self.type_head(emb),
        }
        return {"embeddings": emb, "logits": logits}

    def embedding(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        out = self.forward(x)
        emb = out["embeddings"]
        if normalize:
            emb = nn.functional.normalize(emb, dim=1)
        return emb

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_embedding_model(pretrained: bool = True, **kwargs: Any) -> EmbeddingModel:
    return EmbeddingModel(pretrained=pretrained, **kwargs)
