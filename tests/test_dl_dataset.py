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

from bacteria_assistant.dl.dataset import LabelEncoder, build_image_dataset


@pytest.fixture
def tiny_table(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(6):
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


def test_label_encoder_roundtrip() -> None:
    encoder = LabelEncoder(["a", "b", "a"])
    assert encoder.encode("a") == 0
    assert encoder.encode("b") == 1
    assert encoder.decode(0) == "a"
    assert len(encoder) == 2


def test_build_image_dataset_returns_samples(tiny_table: pd.DataFrame) -> None:
    import torch

    dataset = build_image_dataset(tiny_table)
    assert len(dataset) == 6
    img, _ = dataset[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32


def test_build_image_dataset_targets_are_encoded(tiny_table: pd.DataFrame) -> None:
    dataset = build_image_dataset(tiny_table)
    species_targets = [int(dataset[i][1]["species"]) for i in range(len(dataset))]
    # only 2 distinct organisms -> species indices are 0/1
    assert set(species_targets) <= {0, 1}
    for i in range(len(dataset)):
        targets = dataset[i][1]
        assert set(targets.keys()) == {"species", "group", "gram", "type"}
        assert int(targets["group"]) in {0, 1}
        assert int(targets["gram"]) == 0  # all gram_positive
        assert int(targets["type"]) == 0  # all bacteria


def test_dataset_returns_torch_tensors(tiny_table: pd.DataFrame) -> None:
    import torch

    dataset = build_image_dataset(tiny_table)
    img, _ = dataset[0]
    assert isinstance(img, torch.Tensor)
    # ImageNet-normalized tensors: values exceed [0,1] but stay bounded.
    assert img.shape == (3, 224, 224)
    assert torch.isfinite(img).all()
