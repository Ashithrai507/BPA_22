from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import bacteria_assistant.training as training


class _ConstantPredictor:
    """Deterministic stand-in model so selection tests never depend on sklearn internals."""

    def __init__(self, minority_threshold: float | None = None) -> None:
        self._threshold = minority_threshold

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> _ConstantPredictor:
        return self

    def predict(self, x: pd.DataFrame) -> list[str]:
        if self._threshold is None:
            return ["majority"] * len(x)
        return ["minority" if f1 > self._threshold else "majority" for f1 in x["f1"]]


def _selection_df() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n_maj, n_min = 60, 8
    maj_f1 = rng.uniform(0.0, 10.0, size=n_maj)
    maj_f2 = rng.normal(0.0, 0.5, size=n_maj)
    min_f1 = rng.normal(11.0, 0.3, size=n_min)
    min_f2 = rng.normal(0.0, 0.5, size=n_min)
    x = pd.DataFrame({"f1": np.concatenate([maj_f1, min_f1]), "f2": np.concatenate([maj_f2, min_f2])})
    y = pd.Series(["majority"] * n_maj + ["minority"] * n_min)
    return x, y


def test_oversample_train_raises_minority_to_majority() -> None:
    x = pd.DataFrame({"a": range(10), "b": range(10)})
    y = pd.Series(["maj"] * 6 + ["min_a"] * 3 + ["min_b"] * 1)

    x_res, y_res = training._oversample_train(x, y, random_state=42)

    assert y_res.value_counts().to_dict() == {"maj": 6, "min_a": 6, "min_b": 6}
    assert len(x_res) == len(y_res) == 18
    assert x_res["a"].isin(x["a"]).all()


def test_oversample_train_balanced_input_unchanged() -> None:
    x = pd.DataFrame({"a": range(4)})
    y = pd.Series(["a", "a", "b", "b"])

    x_res, y_res = training._oversample_train(x, y, random_state=42)

    assert list(y_res) == list(y)
    assert list(x_res["a"]) == list(x["a"])


def test_oversample_train_does_not_mutate_input() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    y = pd.Series(["a", "a", "b", "b"])
    x_before = x.copy(deep=True)
    y_before = y.copy(deep=True)

    training._oversample_train(x, y, random_state=42)

    pd.testing.assert_frame_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


def test_per_modality_accuracy() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    modalities = pd.Series(["gram", "gram", "plate", "plate"])

    out = training._per_modality_accuracy(y_true, y_pred, modalities)

    assert out == {"gram": 0.5, "plate": 1.0}


def _split_selection() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x, y = _selection_df()
    return train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)


def test_fit_best_ensemble_records_default_scoring() -> None:
    x_tr, x_te, y_tr, y_te = _split_selection()

    _, summary, _ = training._fit_best_ensemble_model(
        x_tr,
        y_tr,
        x_te,
        y_te,
        random_state=42,
        candidates=[
            ("collapse", _ConstantPredictor()),
            ("minority_aware", _ConstantPredictor(minority_threshold=5.0)),
        ],
    )

    assert summary["scoring_method"] == "accuracy"


def test_fit_best_ensemble_records_balanced_accuracy_scoring() -> None:
    x_tr, x_te, y_tr, y_te = _split_selection()

    _, summary, _ = training._fit_best_ensemble_model(
        x_tr,
        y_tr,
        x_te,
        y_te,
        random_state=42,
        scoring="balanced_accuracy",
        candidates=[
            ("collapse", _ConstantPredictor()),
            ("minority_aware", _ConstantPredictor(minority_threshold=5.0)),
        ],
    )

    assert summary["scoring_method"] == "balanced_accuracy"
    assert 0.0 <= summary["balanced_accuracy"] <= 1.0


def test_fit_best_ensemble_rejects_unknown_scoring() -> None:
    x_tr, x_te, y_tr, y_te = _split_selection()

    with pytest.raises(ValueError):
        training._fit_best_ensemble_model(
            x_tr,
            y_tr,
            x_te,
            y_te,
            random_state=42,
            scoring="not_a_metric",
        )


def test_balanced_accuracy_scoring_changes_candidate_selection() -> None:
    x_tr, x_te, y_tr, y_te = _split_selection()

    _, _, acc_name = training._fit_best_ensemble_model(
        x_tr,
        y_tr,
        x_te,
        y_te,
        random_state=42,
        scoring="accuracy",
        candidates=[
            ("collapse", _ConstantPredictor()),
            ("minority_aware", _ConstantPredictor(minority_threshold=5.0)),
        ],
    )
    _, _, ba_name = training._fit_best_ensemble_model(
        x_tr,
        y_tr,
        x_te,
        y_te,
        random_state=42,
        scoring="balanced_accuracy",
        candidates=[
            ("collapse", _ConstantPredictor()),
            ("minority_aware", _ConstantPredictor(minority_threshold=5.0)),
        ],
    )

    assert acc_name == "collapse"
    assert ba_name == "minority_aware"
