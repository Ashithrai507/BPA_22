from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.training import extract_color_features, train_ensemble


def _make_synthetic_image(tmp_path: Path) -> Path:
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), img)
    return path


def test_extract_color_features_returns_empty_dict_for_missing_path() -> None:
    result = extract_color_features(Path("/nonexistent/image.png"))
    assert result == {}


def test_extract_color_features_returns_expected_keys(tmp_path: Path) -> None:
    img_path = _make_synthetic_image(tmp_path)
    result = extract_color_features(img_path)

    expected_keys = {
        "b_mean", "b_std", "g_mean", "g_std", "r_mean", "r_std",
        "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    }
    assert set(result.keys()) == expected_keys
    assert len(result) == 12


def test_extract_color_features_returns_numeric_values(tmp_path: Path) -> None:
    img_path = _make_synthetic_image(tmp_path)
    result = extract_color_features(img_path)

    for value in result.values():
        assert isinstance(value, (float, np.floating))


def test_train_ensemble_returns_voting_classifier() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((30, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(["A"] * 15 + ["B"] * 15)

    model = train_ensemble(X, y)
    assert isinstance(model, VotingClassifier)


def test_train_ensemble_can_predict() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((30, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(["A"] * 15 + ["B"] * 15)

    model = train_ensemble(X, y)
    preds = model.predict(X)
    assert len(preds) == 30
    assert all(p in ("A", "B") for p in preds)


def test_extract_color_features_returns_dict(tmp_path: Path) -> None:
    img_path = _make_synthetic_image(tmp_path)
    result = extract_color_features(img_path)
    assert isinstance(result, dict)
