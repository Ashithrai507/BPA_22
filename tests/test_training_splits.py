from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.training import _image_level_split


def _synthetic_labeled_df(n_per_shape: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_per_shape):
        for shape in ("cocci", "bacilli", "fungal"):
            rows.append(
                {
                    "image_path": f"data/img_{i}_{shape}.png",
                    "organism": f"organism_{shape}",
                    "shape_label": shape,
                }
            )
    return pd.DataFrame(rows)


def test_image_level_split_has_no_image_overlap() -> None:
    df = _synthetic_labeled_df()
    train_df, test_df = _image_level_split(df, random_state=42)

    train_paths = set(train_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert train_paths.isdisjoint(test_paths)
    assert len(train_df) + len(test_df) == len(df)


def test_image_level_split_preserves_all_shape_classes_in_train() -> None:
    df = _synthetic_labeled_df()
    train_df, _ = _image_level_split(df, random_state=42)

    assert set(train_df["shape_label"]) == {"cocci", "bacilli", "fungal"}


def test_image_level_split_falls_back_when_stratify_impossible() -> None:
    df = _synthetic_labeled_df()
    single_fungal = df[df["shape_label"] == "fungal"].head(1)
    df = pd.concat([df[df["shape_label"] != "fungal"], single_fungal])

    train_df, test_df = _image_level_split(df, random_state=42)

    assert set(train_df["image_path"]).isdisjoint(set(test_df["image_path"]))
    assert len(train_df) + len(test_df) == len(df)
