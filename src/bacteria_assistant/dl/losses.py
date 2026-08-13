from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) over a batch.

    Positive pairs are samples sharing the same label; negatives share different
    labels. The normalized cosine similarities are used, so embeddings should be
    L2-normalized before calling.
    """
    embeddings = F.normalize(embeddings, dim=1)
    n = embeddings.size(0)
    labels = labels.view(-1)
    sim = embeddings @ embeddings.T / temperature

    # Mask out self-similarity.
    eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_label = same_label & ~eye

    logsumexp = torch.logsumexp(sim, dim=1, keepdim=True)
    positive = sim * same_label.float()
    positive = positive.masked_fill(~same_label, 0.0)
    log_prob = positive.sum(dim=1) - logsumexp.squeeze(1)

    pos_counts = same_label.float().sum(dim=1)
    loss = -(log_prob / pos_counts.clamp(min=1)).mean()
    return loss


class CombinedLoss(nn.Module):
    """Weighted sum of the species CE plus auxiliary head CEs and contrastive loss."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
        head_keys: tuple[str, ...] = ("group", "gram", "type"),
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.head_keys = head_keys

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        species = F.cross_entropy(logits["species"], targets["species"])
        parts: dict[str, torch.Tensor] = {"species": species}

        aux = torch.zeros((), device=species.device)
        for key in self.head_keys:
            loss = F.cross_entropy(logits[key], targets[key])
            parts[key] = loss
            aux = aux + loss

        contrastive = torch.zeros((), device=species.device)
        if self.contrastive_weight > 0 and embeddings.size(0) > 1:
            contrastive = supervised_contrastive_loss(
                embeddings,
                targets["species"],
                temperature=self.temperature,
            )
        parts["contrastive"] = contrastive

        total = species + self.aux_weight * aux + self.contrastive_weight * contrastive
        return total, parts
