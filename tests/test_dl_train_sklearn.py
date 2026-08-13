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

import cv2

from bacteria_assistant.dl.model import EmbeddingModel
from bacteria_assistant.training import _build_image_embedding_table


@pytest.fixture
def labeled_df(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(4):
        img_path = tmp_path / f"img_{i}.png"
        np_img = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(img_path), np_img)
        rows.append(
            {
                "image_path": str(img_path),
                "organism": "Staphylococcus aureus" if i % 2 else "Bacillus subtilis",
                "organism_type": "bacteria",
                "gram_label": "gram_positive",
                "shape_label": "cocci",
                "taxonomy_group": "gram_positive_cocci" if i % 2 else "gram_positive_bacilli",
                "imaging_type": "gram stain",
            }
        )
    return pd.DataFrame(rows)


def test_build_image_embedding_table_has_emb_columns(labeled_df: pd.DataFrame, tmp_path: Path) -> None:
    model = EmbeddingModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_species=2,
        num_groups=2,
        num_grams=2,
        num_types=2,
    )
    model.eval()
    table = _build_image_embedding_table(labeled_df, model, embedding_dim=1280, workspace_root=tmp_path)
    emb_cols = [c for c in table.columns if c.startswith("emb_")]
    assert len(emb_cols) == 1280
    assert len(table) == 4
    assert "organism" in table.columns
    assert table["organism"].iloc[0] == "Bacillus subtilis"


def test_build_image_embedding_table_values_finite(labeled_df: pd.DataFrame, tmp_path: Path) -> None:
    model = EmbeddingModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_species=2,
        num_groups=2,
        num_grams=2,
        num_types=2,
    )
    model.eval()
    table = _build_image_embedding_table(labeled_df, model, embedding_dim=1280, workspace_root=tmp_path)
    emb_cols = [c for c in table.columns if c.startswith("emb_")]
    assert np.isfinite(table[emb_cols].to_numpy()).all()
