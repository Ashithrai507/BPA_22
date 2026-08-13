from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2

from bacteria_assistant.dl.extract import predict_species
from bacteria_assistant.dl.model import EmbeddingModel


@pytest.fixture(scope="module")
def model_and_encoder() -> tuple[EmbeddingModel, object]:
    model = EmbeddingModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_species=3,
        num_groups=2,
        num_grams=2,
        num_types=2,
    )
    model.eval()

    class DummyEncoder:
        def decode(self, index: int) -> str:
            return f"species_{index}"

        def __len__(self) -> int:
            return 3

    return model, DummyEncoder()


def _probe_image(tmp_path: Path) -> Path:
    img = (np.random.default_rng(7).random((96, 96, 3)) * 255).astype(np.uint8)
    p = tmp_path / "probe.png"
    cv2.imwrite(str(p), img)
    return p


def test_predict_species_returns_name_and_confidence(
    model_and_encoder: tuple[EmbeddingModel, object], tmp_path: Path
) -> None:
    model, encoder = model_and_encoder
    name, conf = predict_species(model, _probe_image(tmp_path), encoder, input_size=224)
    assert isinstance(name, str)
    assert 0.0 <= conf <= 1.0


def test_predict_species_tta_matches_plain(model_and_encoder: tuple[EmbeddingModel, object], tmp_path: Path) -> None:
    model, encoder = model_and_encoder
    p = _probe_image(tmp_path)
    name_plain, conf_plain = predict_species(model, p, encoder, input_size=224, tta=True)
    name_plain2, _ = predict_species(model, p, encoder, input_size=224, tta=False)
    assert name_plain == name_plain2
    assert 0.0 <= conf_plain <= 1.0
