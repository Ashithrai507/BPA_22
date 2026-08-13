from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.dl.model import EmbeddingModel


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    return EmbeddingModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_species=10,
        num_groups=4,
        num_grams=3,
        num_types=2,
    )


def test_embedding_dim_is_1280_for_efficientnet_b0(model: EmbeddingModel) -> None:
    assert model.embedding_dim == 1280


def test_embedding_shape(model: EmbeddingModel) -> None:
    x = torch.randn(2, 3, 224, 224)
    emb = model.embedding(x)
    assert emb.shape == (2, 1280)
    assert emb.dtype == torch.float32


def test_normalized_embedding_has_unit_norm(model: EmbeddingModel) -> None:
    x = torch.randn(2, 3, 224, 224)
    emb = model.embedding(x, normalize=True)
    norms = torch.linalg.vector_norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_forward_returns_all_logits(model: EmbeddingModel) -> None:
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert set(out["logits"].keys()) == {"species", "group", "gram", "type"}
    assert out["logits"]["species"].shape == (2, 10)
    assert out["logits"]["group"].shape == (2, 4)
    assert out["logits"]["gram"].shape == (2, 3)
    assert out["logits"]["type"].shape == (2, 2)
    assert out["embeddings"].shape == (2, 1280)


def test_backbone_freeze_controls_gradients(model: EmbeddingModel) -> None:
    model.freeze_backbone()
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    assert backbone_params
    assert all(not p.requires_grad for p in backbone_params)

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in backbone_params)
