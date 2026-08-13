from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.dl.losses import (
    CombinedLoss,
    supervised_contrastive_loss,
)


def test_supervised_contrastive_loss_identical_samples_is_zero() -> None:
    # Positives identical and opposite to negatives -> loss ~ 0.
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(emb, labels, temperature=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_supervised_contrastive_loss_different_samples_is_positive() -> None:
    # Two identical clusters but pushed apart -> loss > 0.
    emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(emb, labels, temperature=0.5)
    assert loss.item() > 0.0


def test_combined_loss_returns_species_loss_when_weights_zero() -> None:
    logits = {
        "species": torch.randn(4, 10),
        "group": torch.randn(4, 4),
        "gram": torch.randn(4, 3),
        "type": torch.randn(4, 2),
    }
    targets = {
        "species": torch.randint(0, 10, (4,)),
        "group": torch.randint(0, 4, (4,)),
        "gram": torch.randint(0, 3, (4,)),
        "type": torch.randint(0, 2, (4,)),
    }
    emb = torch.randn(4, 1280)
    criterion = CombinedLoss(aux_weight=0.0, contrastive_weight=0.0)
    total, parts = criterion(logits, targets, emb)

    expected_species = torch.nn.functional.cross_entropy(logits["species"], targets["species"])
    assert total.item() == pytest.approx(expected_species.item())
    assert parts["species"].item() == pytest.approx(expected_species.item())
    assert parts["contrastive"].item() == 0.0


def test_combined_loss_scalar_output() -> None:
    logits = {
        "species": torch.randn(4, 10),
        "group": torch.randn(4, 4),
        "gram": torch.randn(4, 3),
        "type": torch.randn(4, 2),
    }
    targets = {
        "species": torch.randint(0, 10, (4,)),
        "group": torch.randint(0, 4, (4,)),
        "gram": torch.randint(0, 3, (4,)),
        "type": torch.randint(0, 2, (4,)),
    }
    emb = torch.randn(4, 1280)
    criterion = CombinedLoss(aux_weight=0.3, contrastive_weight=0.1)
    total, _ = criterion(logits, targets, emb)
    assert total.dim() == 0
    assert total.item() > 0.0
