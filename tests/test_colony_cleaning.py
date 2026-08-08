from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import bacteria_assistant.training as training


def _colony_table(**overrides: list) -> pd.DataFrame:
    base = {
        "image_id": ["img_a", "img_a", "img_b", "img_b", "img_c", "img_d"],
        "shape_label": ["cocci", "cocci", "bacilli", "bacilli", "fungal", "spiral"],
        "aspect_ratio": [1.1, 3.0, 1.8, 1.05, 1.4, 2.5],
        "solidity": [0.97, 0.98, 0.96, 0.93, 0.99, 0.9],
        "imaging_type": ["gram", "gram", "plate", "plate", "gram", "plate"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_colony_label_table_removes_contradicting_rows() -> None:
    df = _colony_table()

    cleaned, stats = training._clean_colony_label_table(df)

    # Per-row semantics: img_a/img_b keep their valid rows; img_c's only
    # (contradicting) row is dropped; img_d (spiral, no rule) is untouched.
    assert set(cleaned["image_id"]) == {"img_a", "img_b", "img_d"}
    assert set(cleaned["shape_label"]) == {"cocci", "bacilli", "spiral"}


def test_clean_colony_label_table_removal_counts_per_shape() -> None:
    cleaned, stats = training._clean_colony_label_table(_colony_table())

    assert stats["colony_rows_before_cleaning"] == 6
    assert stats["colony_rows_removed_by_cleaning"] == 3
    assert stats["colony_cleaning_removals"] == {
        "cocci": 1,
        "bacilli": 1,
        "fungal": 1,
        "spiral": 0,
    }
    assert len(cleaned) == 3


def test_clean_colony_label_table_is_noop_when_nothing_contradicts() -> None:
    df = _colony_table(
        shape_label=["cocci", "cocci", "bacilli", "bacilli", "fungal", "spiral"],
        aspect_ratio=[1.1, 1.9, 1.8, 1.6, 1.4, 2.5],
        solidity=[0.9, 0.88, 0.96, 0.93, 0.8, 0.9],
    )

    cleaned, stats = training._clean_colony_label_table(df)

    assert len(cleaned) == len(df)
    assert stats["colony_rows_removed_by_cleaning"] == 0
    assert all(count == 0 for count in stats["colony_cleaning_removals"].values())


def test_clean_colony_label_table_does_not_mutate_input() -> None:
    df = _colony_table()
    before = df.copy(deep=True)

    training._clean_colony_label_table(df)

    pd.testing.assert_frame_equal(df, before)


def test_clean_colony_label_table_empty_frame() -> None:
    cleaned, stats = training._clean_colony_label_table(pd.DataFrame(columns=["shape_label", "aspect_ratio", "solidity"]))

    assert cleaned.empty
    assert stats["colony_rows_before_cleaning"] == 0
    assert stats["colony_rows_removed_by_cleaning"] == 0
