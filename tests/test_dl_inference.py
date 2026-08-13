from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from sklearn.dummy import DummyClassifier

from bacteria_assistant.config import DATASET_ROOT
from bacteria_assistant.dl.model import EmbeddingModel
from bacteria_assistant.inference import load_models, predict_bacteria_image

COLONY_FEATURES = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "solidity",
    "equivalent_diameter",
    "mean_intensity",
]


def _first_image() -> Path:
    dataset_roots = [PROJECT_ROOT / DATASET_ROOT, PROJECT_ROOT / "Bacteria dataset"]
    for dataset_root in dataset_roots:
        if dataset_root.exists():
            for path in dataset_root.rglob("*.png"):
                return path
    raise FileNotFoundError("No PNG image found in dataset.")


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "embedding_model.pt"
    model = EmbeddingModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_species=2,
        num_groups=2,
        num_grams=2,
        num_types=2,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_encoders": {
                "species": {"labels": ["a", "b"]},
                "group": {"labels": ["x", "y"]},
                "gram": {"labels": ["u", "v"]},
                "type": {"labels": ["z", "q"]},
            },
            "config": {"backbone": "efficientnet_b0", "embedding_dim": 1280},
        },
        ckpt,
    )
    return ckpt


@pytest.fixture
def fake_artifact(tmp_path: Path, fake_checkpoint: Path) -> Path:
    artifact_path = tmp_path / "bacteria_models.joblib"
    embed_cols = [f"emb_{i}" for i in range(1280)]

    def clf(labels: list[str]) -> DummyClassifier:
        import numpy as np

        model = DummyClassifier(strategy="most_frequent")
        model.fit(np.zeros((4, 1280)), [labels[i % len(labels)] for i in range(4)])
        return model

    joblib.dump(
        {
            "image_feature_columns": embed_cols,
            "colony_feature_columns": COLONY_FEATURES,
            "embedding_model_path": str(fake_checkpoint),
            "organism_type_model": clf(["bacteria", "fungi"]),
            "group_model": clf(["g0", "g1"]),
            "organism_model": clf(["o0", "o1"]),
            "gram_model": clf(["gram_positive", "gram_negative"]),
            "shape_model": clf(["cocci", "bacilli", "spiral"]),
            "organism_metadata": {},
        },
        artifact_path,
    )
    return artifact_path


def test_load_models_attaches_embedding_model(fake_artifact: Path, fake_checkpoint: Path) -> None:
    artifacts = load_models(fake_artifact)
    assert artifacts["embedding_model_path"] == str(fake_checkpoint)
    assert "_embedding_model" in artifacts


def test_load_models_skips_embedding_when_absent(tmp_path: Path, fake_checkpoint: Path) -> None:
    artifact_path = tmp_path / "classical.joblib"
    joblib.dump(
        {
            "image_feature_columns": ["f0", "f1"],
            "colony_feature_columns": COLONY_FEATURES,
            "organism_type_model": DummyClassifier(),
            "group_model": DummyClassifier(),
            "organism_model": DummyClassifier(),
            "gram_model": DummyClassifier(),
            "shape_model": DummyClassifier(),
            "organism_metadata": {},
        },
        artifact_path,
    )
    artifacts = load_models(artifact_path)
    assert "_embedding_model" not in artifacts


def test_predict_uses_embedding_columns(fake_artifact: Path) -> None:
    output = predict_bacteria_image(_first_image(), model_path=fake_artifact, mode="basic")
    assert output["organism_type"] in {"bacteria", "fungi"}
    assert output["predicted_bacteria_name"]
    assert output["bacteria_type"] in {
        "gram_positive",
        "gram_negative",
        "non_bacterial_fungi",
    }
    assert output["dominant_shape"] in {"cocci", "bacilli", "spiral", "fungal"}
