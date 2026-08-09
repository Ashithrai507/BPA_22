from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

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
    empty = pd.DataFrame(columns=["shape_label", "aspect_ratio", "solidity"])
    cleaned, stats = training._clean_colony_label_table(empty)

    assert cleaned.empty
    assert stats["colony_rows_before_cleaning"] == 0
    assert stats["colony_rows_removed_by_cleaning"] == 0
    assert stats["colony_cleaning_removals"] == {
        "cocci": 0,
        "bacilli": 0,
        "spiral": 0,
        "fungal": 0,
    }


def test_train_models_records_colony_cleaning_meta(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "organism": ["Staphylococcus aureus", "Staphylococcus aureus"],
            "image_path": ["img_a.png", "img_b.png"],
            "imaging_type": ["gram", "gram"],
        }
    ).to_csv(csv_path, index=False)

    colony_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "image_id": ["img_a", "img_b", "img_c", "img_d", "img_e", "img_f"],
                    "shape_label": ["cocci", "cocci", "bacilli", "bacilli", "fungal", "spiral"],
                    "aspect_ratio": [1.1, 3.0, 1.8, 1.05, 1.4, 2.5],
                    "solidity": [0.97, 0.98, 0.96, 0.93, 0.99, 0.9],
                    "imaging_type": ["gram", "gram", "plate", "plate", "gram", "plate"],
                }
            )
        ]
        * 5,
        ignore_index=True,
    )
    monkeypatch.setattr(training, "_build_colony_feature_table", lambda labeled_df, workspace_root: colony_table)
    monkeypatch.setattr(
        training,
        "_build_image_feature_table",
        lambda labeled_df, workspace_root, *args, **kwargs: pd.DataFrame(
            {
                "image_path": ["img_a.png", "img_b.png"],
                "organism": ["Staphylococcus aureus", "Staphylococcus aureus"],
                "organism_type": ["bacteria", "bacteria"],
                "gram_label": ["gram_positive", "gram_positive"],
                "shape_label": ["cocci", "cocci"],
                "taxonomy_group": ["gram_positive_cocci", "gram_positive_cocci"],
                "imaging_type": ["gram", "gram"],
                "r_mean": [0.5, 0.6],
            }
        ),
    )
    monkeypatch.setattr(
        training,
        "_fit_best_ensemble_model",
        lambda x_train, y_train, *a, **k: (
            DummyClassifier().fit(x_train, y_train),
            {"accuracy": 1.0},
            "constant",
        ),
    )

    metrics = training.train_models(
        dataset_csv=csv_path,
        workspace_root=tmp_path,
        model_output_path=tmp_path / "models.joblib",
        random_state=42,
    )

    meta = metrics["training_meta"]
    assert meta["colony_rows_before_cleaning"] == 30
    assert meta["colony_rows_removed_by_cleaning"] == 15
    assert meta["colony_cleaning_removals"]["cocci"] == 5
    assert meta["colony_cleaning_removals"]["bacilli"] == 5
    assert meta["colony_cleaning_removals"]["fungal"] == 5
    assert meta["colony_cleaning_removals"]["spiral"] == 0


def test_split_colonies_by_image_no_shared_images() -> None:
    rng = np.random.default_rng(0)
    image_ids = [f"img_{i}" for i in range(50)]
    rows = []
    for img in image_ids:
        for _ in range(4):
            rows.append(
                {
                    "image_id": img,
                    "shape_label": "cocci" if rng.random() < 0.5 else "bacilli",
                    "aspect_ratio": 1.0,
                    "solidity": 0.9,
                    "imaging_type": "gram",
                }
            )
    colony_table = pd.DataFrame(rows)

    train_idx, test_idx = training._split_colonies_by_image(colony_table, random_state=42)

    train_images = set(colony_table.loc[train_idx, "image_id"])
    test_images = set(colony_table.loc[test_idx, "image_id"])
    assert train_images.isdisjoint(test_images)
    assert len(train_idx) > 0 and len(test_idx) > 0
    assert len(train_idx) + len(test_idx) == len(colony_table)


def test_split_colonies_by_image_reproducible() -> None:
    colony_table = _colony_table()
    colony_table = pd.concat([colony_table] * 3, ignore_index=True)

    a_train, a_test = training._split_colonies_by_image(colony_table, random_state=7)
    b_train, b_test = training._split_colonies_by_image(colony_table, random_state=7)

    assert a_train.equals(b_train)
    assert a_test.equals(b_test)


def test_split_colonies_by_image_raises_when_no_test_split_possible() -> None:
    colony_table = _colony_table()
    single = colony_table.iloc[[0]].copy()

    with pytest.raises(ValueError):
        training._split_colonies_by_image(single, test_size=0.2, random_state=42)
