from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import (
    AUGMENT_PER_IMAGE,
    CLAHE_CLIP_LIMIT,
    FEATURE_VERSION,
)
from bacteria_assistant.features import (
    augment_image,
    extract_colonies,
    extract_image_features,
)
from bacteria_assistant.training import _per_modality_metrics, _resolve_image_path


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def test_clahe_feature_extraction_runs() -> None:
    feats = extract_image_features(_synthetic_image())
    assert "gray_spatial_0" in feats
    assert "gray_spatial_575" in feats
    assert 0.0 <= feats["gray_spatial_0"] <= 1.0


def test_clahe_segmentation_runs() -> None:
    colonies = extract_colonies(_synthetic_image())
    assert isinstance(colonies, list)


def test_augment_image_variant_count() -> None:
    variants = augment_image(_synthetic_image(), seed=1)
    assert len(variants) == AUGMENT_PER_IMAGE
    for variant in variants:
        assert variant.shape == (256, 256, 3)
        assert variant.dtype == np.uint8


def test_augmentation_is_deterministic_per_seed() -> None:
    img = _synthetic_image()
    a = augment_image(img, seed=42)
    b = augment_image(img, seed=42)
    for va, vb in zip(a, b, strict=True):
        np.testing.assert_array_equal(va, vb)


def test_resolve_image_path_fallback() -> None:
    if not (PROJECT_ROOT / "data" / "dataset").exists():
        return
    sample = next((PROJECT_ROOT / "data" / "dataset").rglob("*.png"))
    csv_style = str(sample.relative_to(PROJECT_ROOT)).replace("data/dataset/", "Bacteria dataset/", 1)
    resolved = _resolve_image_path(PROJECT_ROOT, csv_style)
    assert resolved.exists()


def test_per_modality_metrics_structure() -> None:
    import pandas as pd

    table = pd.DataFrame(
        {
            "imaging_type": ["gram stain", "gram stain", "media plate", "media plate"],
            "label": ["a", "a", "b", "b"],
        }
    )
    preds = ["a", "b", "b", "b"]
    result = _per_modality_metrics(table, "label", preds)
    assert set(result.keys()) == {"gram stain", "media plate"}
    assert result["gram stain"]["n"] == 2
    assert result["gram stain"]["accuracy"] == 0.5
    assert result["media plate"]["n"] == 2
    assert result["media plate"]["accuracy"] == 1.0


def test_config_feature_version() -> None:
    assert isinstance(FEATURE_VERSION, int)
    assert FEATURE_VERSION > 0
    assert CLAHE_CLIP_LIMIT > 0
