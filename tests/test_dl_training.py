from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn

from bacteria_assistant.dl.dataset import LabelEncoder, build_image_dataset
from bacteria_assistant.dl.trainer import TrainingConfig, fit_embedding_model


class TinyBackbone(nn.Module):
    """Minimal backbone producing a 16-dim embedding for fast smoke tests."""

    def __init__(self, embedding_dim: int = 16, num_species: int = 2) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = "tiny"
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(8, embedding_dim)
        self.heads = {
            "species": nn.Linear(embedding_dim, num_species),
            "group": nn.Linear(embedding_dim, 2),
            "gram": nn.Linear(embedding_dim, 1),
            "type": nn.Linear(embedding_dim, 1),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.conv(x).flatten(1)
        emb = self.embedding(h)
        return {
            "embeddings": torch.nn.functional.normalize(emb, dim=1),
            "logits": {
                "species": self.heads["species"](emb),
                "group": self.heads["group"](emb),
                "gram": self.heads["gram"](emb),
                "type": self.heads["type"](emb),
            },
        }


@pytest.fixture
def tiny_table(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(8):
        img_path = tmp_path / f"img_{i}.png"
        np_img = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype(np.uint8)
        import cv2

        cv2.imwrite(str(img_path), np_img)
        rows.append(
            {
                "image_path": str(img_path),
                "organism": "Staphylococcus aureus" if i % 2 else "Bacillus subtilis",
                "organism_type": "bacteria",
                "gram_label": "gram_positive",
                "taxonomy_group": "gram_positive_cocci" if i % 2 else "gram_positive_bacilli",
            }
        )
    return pd.DataFrame(rows)


def test_fit_embedding_model_reduces_loss(tiny_table: pd.DataFrame, tmp_path: Path) -> None:
    dataset = build_image_dataset(tiny_table)
    model = TinyBackbone(num_species=2)

    config = TrainingConfig(
        batch_size=4,
        epochs=2,
        lr=1e-3,
        aux_weight=0.3,
        contrastive_weight=0.1,
        num_workers=0,
        seed=0,
    )
    history, _ = fit_embedding_model(
        model=model,
        dataset=dataset,
        config=config,
        checkpoint_path=tmp_path / "ckpt.pt",
        device="cpu",
    )
    assert len(history["loss"]) == 2
    assert history["loss"][-1] <= history["loss"][0]
    assert tmp_path.joinpath("ckpt.pt").exists()


def test_fit_embedding_model_saves_checkpoint_with_encoders(tiny_table: pd.DataFrame, tmp_path: Path) -> None:
    dataset = build_image_dataset(tiny_table)
    model = TinyBackbone(num_species=2)
    config = TrainingConfig(
        batch_size=4,
        epochs=1,
        lr=1e-3,
        num_workers=0,
        seed=0,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    _, metadata = fit_embedding_model(
        model=model,
        dataset=dataset,
        config=config,
        checkpoint_path=ckpt_path,
        device="cpu",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = LabelEncoder(ckpt["label_encoders"]["species"]["labels"])
    assert enc.encode("Bacillus subtilis") == 0
    assert ckpt["config"]["embedding_dim"] == 16
    assert metadata["best_val_acc_species"] >= 0.0
