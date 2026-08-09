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

import bacteria_assistant.training as training
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


def _imbalanced_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "organism": ["Bacillus subtilis"] * 4 + ["Staphylococcus aureus"] * 16,
            "image_path": [f"img_{i}.png" for i in range(20)],
            "imaging_type": (
                ["gram stain"] * 2
                + ["media plate"] * 2
                + ["gram stain"] * 12
                + ["media plate"] * 4
            ),
        }
    )


def test_balance_modalities_equalizes_imbalanced_species() -> None:
    balanced, stats = training._balance_modalities_per_species(_imbalanced_train_df(), random_state=42)

    aureus = balanced[balanced["organism"] == "Staphylococcus aureus"]
    assert aureus["imaging_type"].value_counts().to_dict() == {"gram stain": 4, "media plate": 4}
    assert stats["Staphylococcus aureus"]["balanced"] is True
    assert stats["Staphylococcus aureus"]["before"] == {"gram stain": 12, "media plate": 4}
    assert stats["Staphylococcus aureus"]["after"] == {"gram stain": 4, "media plate": 4}


def test_balance_modalities_leaves_balanced_species_untouched() -> None:
    balanced, stats = training._balance_modalities_per_species(_imbalanced_train_df(), random_state=42)

    subtilis = balanced[balanced["organism"] == "Bacillus subtilis"]
    assert subtilis["imaging_type"].value_counts().to_dict() == {"gram stain": 2, "media plate": 2}
    assert stats["Bacillus subtilis"]["after"] == {"gram stain": 2, "media plate": 2}


def test_balance_modalities_reproducible() -> None:
    a, _ = training._balance_modalities_per_species(_imbalanced_train_df(), random_state=7)
    b, _ = training._balance_modalities_per_species(_imbalanced_train_df(), random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_balance_modalities_does_not_mutate_input() -> None:
    df = _imbalanced_train_df()
    before = df.copy(deep=True)
    training._balance_modalities_per_species(df, random_state=42)
    pd.testing.assert_frame_equal(df, before)


def test_balance_modalities_single_modality_species_untouched() -> None:
    df = pd.DataFrame(
        {
            "organism": ["Candida albicans"] * 4,
            "image_path": [f"img_{i}.png" for i in range(4)],
            "imaging_type": ["gram stain"] * 4,
        }
    )
    balanced, stats = training._balance_modalities_per_species(df, random_state=42)
    assert len(balanced) == 4
    assert stats["Candida albicans"]["balanced"] is False


def test_balance_modalities_empty_frame() -> None:
    balanced, stats = training._balance_modalities_per_species(pd.DataFrame(), random_state=42)
    assert balanced.empty
    assert stats == {}
