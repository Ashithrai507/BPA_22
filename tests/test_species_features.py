from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.features import (
    _colony_shape_features,
    _gradient_texture_features,
    _hu_log_scale,
    extract_image_features,
)


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def _synthetic_gray(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)


def test_hu_log_scale_preserves_sign_and_scales_magnitude() -> None:
    scaled = _hu_log_scale(np.array([0.0, 1.0, -2.0]))
    assert scaled[0] == 0.0
    assert scaled[1] == pytest.approx(np.log1p(1.0))
    assert scaled[2] == pytest.approx(-np.log1p(2.0))


def test_colony_shape_features_include_new_keys() -> None:
    feats = _colony_shape_features(_synthetic_gray())
    for i in range(7):
        assert f"hu_{i}" in feats
        assert f"colony_hu_mean_{i}" in feats
        assert f"colony_hu_std_{i}" in feats
    assert "colony_count" in feats
    assert feats["colony_count"] >= 0


def test_colony_shape_features_empty_mask_returns_zeros() -> None:
    feats = _colony_shape_features(np.zeros((256, 256), dtype=np.uint8))
    assert feats["colony_count"] == 0
    for i in range(7):
        assert feats[f"hu_{i}"] == 0.0
        assert feats[f"colony_hu_mean_{i}"] == 0.0
        assert feats[f"colony_hu_std_{i}"] == 0.0


def test_colony_shape_features_deterministic() -> None:
    a = _colony_shape_features(_synthetic_gray(seed=1))
    b = _colony_shape_features(_synthetic_gray(seed=1))
    assert a == b


def test_gradient_texture_features_finite() -> None:
    feats = _gradient_texture_features(_synthetic_gray())
    assert feats["grad_mag_mean"] >= 0.0
    assert feats["grad_mag_std"] >= 0.0
    assert np.isfinite(feats["grad_angle_std"])


def test_extract_image_features_includes_species_features() -> None:
    feats = extract_image_features(_synthetic_image())
    for key in (
        "hu_0",
        "colony_count",
        "colony_hu_mean_0",
        "colony_hu_std_0",
        "grad_mag_mean",
        "grad_mag_std",
        "grad_angle_std",
    ):
        assert key in feats
