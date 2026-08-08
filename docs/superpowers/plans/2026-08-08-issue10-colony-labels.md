# Issue #10 + #6: Noisy Colony Labels & Image-Level Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove label noise from colony shape training (drop rows whose geometry contradicts the parent image label) and stop plates from leaking across the shape train/test split, raising shape accuracy from 0.675 toward/above 0.72.

**Architecture:** A pure DataFrame filter (`_clean_colony_label_table`) applies conservative per-shape rules from `config.py` and reports removal counts for `training_meta`. `_build_colony_feature_table` emits an `image_id`; a new `_split_colonies_by_image` uses `GroupShuffleSplit` so no image straddles folds. Both changes live in `src/bacteria_assistant/training.py` and are covered by new unit tests.

**Tech Stack:** Python 3.11, pandas, scikit-learn (`GroupShuffleSplit`), pytest, ruff.

## Global Constraints

- Line length 120, target py311 (`pyproject.toml` ruff config). No new runtime dependencies.
- Windows local shell; no `make`/`ruff` on PATH. Run tests via `python -m pytest -q`; install ruff via `python -m pip install ruff==0.6.9`.
- Dataset lives at `dataset/` (NOT `data/dataset`); the local dataset folder must stay out of git (add `dataset/` to `.gitignore`).
- `make train` equivalent: `python scripts/train_model.py --dataset-csv dataset/dataset_full.csv --workspace-root D:\BPA_22` (writes `artifacts/bacteria_models.joblib` + `.metrics.json`).
- Baseline shape accuracy to beat: 0.675 (acceptance criterion >= 0.72).
- Existing lint/test commands are `ruff check .` + `ruff format --check .` and `python -m pytest -q`.

---

### Task 1: Config thresholds and cleaning filter with unit tests

**Files:**
- Modify: `src/bacteria_assistant/config.py` (after `UNKNOWN_LABEL`, before the rejection thresholds)
- Create: `tests/test_colony_cleaning.py`

**Interfaces:**
- Consumes: `SHAPE_CLEANING_RULES` (defined in this task)
- Produces: `training._clean_colony_label_table(colony_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_colony_cleaning.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_colony_cleaning.py -q`
Expected: FAIL with `AttributeError: module 'bacteria_assistant.training' has no attribute '_clean_colony_label_table'` (or `SHAPE_CLEANING_RULES` not found if imported).

- [ ] **Step 3: Add config rules**

In `src/bacteria_assistant/config.py`, after `UNKNOWN_LABEL = "unknown"`:

```python
# Heuristic cleaning for colony shape training labels (issue #10).
# A colony row is dropped when its geometry strongly contradicts the parent
# image's shape_label. Thresholds are deliberately conservative; tune with
# `make evaluate` on the shape model.
SHAPE_CLEANING_RULES = {
    "cocci": {"max_aspect_ratio": 2.0},
    "bacilli": {"min_aspect_ratio": 1.2},
    "fungal": {"max_solidity": 0.95},
}
```

- [ ] **Step 4: Implement the cleaning filter**

In `src/bacteria_assistant/training.py`:
- Update the `.config` import to: `from .config import MODEL_PATH, ORGANISM_METADATA, SUPPORTED_SHAPES, SHAPE_CLEANING_RULES, normalize_organism_name`
- Add this function after `_build_colony_feature_table`:

```python
def _clean_colony_label_table(colony_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop colony rows whose geometry contradicts their shape_label (issue #10).

    Returns the cleaned table and removal statistics for `training_meta`.
    Removal counts are emitted for every supported shape so the audit trail is
    complete (shapes without a rule always report 0).
    """
    removals: dict[str, int] = {shape: 0 for shape in SUPPORTED_SHAPES}
    if colony_table.empty:
        stats = {
            "colony_rows_before_cleaning": 0,
            "colony_rows_removed_by_cleaning": 0,
            "colony_cleaning_removals": removals,
        }
        return colony_table.copy(), stats

    keep = pd.Series(True, index=colony_table.index)
    for shape_label, rules in SHAPE_CLEANING_RULES.items():
        contradict = colony_table["shape_label"] == shape_label
        if "max_aspect_ratio" in rules:
            contradict &= colony_table["aspect_ratio"] > rules["max_aspect_ratio"]
        if "min_aspect_ratio" in rules:
            contradict &= colony_table["aspect_ratio"] < rules["min_aspect_ratio"]
        if "max_solidity" in rules:
            contradict &= colony_table["solidity"] > rules["max_solidity"]
        removals[shape_label] = int(contradict.sum())
        keep &= ~contradict

    cleaned = colony_table.loc[keep].copy()
    stats = {
        "colony_rows_before_cleaning": int(len(colony_table)),
        "colony_rows_removed_by_cleaning": int((~keep).sum()),
        "colony_cleaning_removals": removals,
    }
    return cleaned, stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_colony_cleaning.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_assistant/config.py src/bacteria_assistant/training.py tests/test_colony_cleaning.py
git commit -m "feat(training): clean colony rows contradicting shape label"
```

---

### Task 2: Wire cleaning into train_models and training_meta

**Files:**
- Modify: `src/bacteria_assistant/training.py` (`train_models`, around lines 424-425 and 461-470)

**Interfaces:**
- Consumes: `_clean_colony_label_table(colony_table) -> tuple[pd.DataFrame, dict[str, int]]` (Task 1)
- Produces: `training_meta` keys `colony_rows_before_cleaning`, `colony_rows_removed_by_cleaning`, `colony_cleaning_removals` in the artifacts/metrics JSON.

- [ ] **Step 1: Write the failing test**

In `tests/test_training_quality.py`, append:

```python
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
    monkeypatch.setattr(training, "_build_image_feature_table", lambda labeled_df, workspace_root: pd.DataFrame(
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
    ))
    monkeypatch.setattr(training, "_fit_best_ensemble_model", lambda *a, **k: (_ConstantPredictor(), {"accuracy": 1.0}, "constant"))

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
```

Note: `_ConstantPredictor` is already defined at the top of `test_training_quality.py`. Import `training` and `pandas` at the top of `tests/test_training_quality.py` (they are already imported there).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_quality.py::test_train_models_records_colony_cleaning_meta -q`
Expected: FAIL with `KeyError: 'colony_rows_before_cleaning'`

- [ ] **Step 3: Modify train_models**

In `src/bacteria_assistant/training.py`, replace:

```python
    colony_table = _build_colony_feature_table(labeled_df, workspace_root)
    colony_feature_cols = [c for c in colony_table.columns if c not in {"shape_label", "imaging_type"}]
```

with:

```python
    colony_table = _build_colony_feature_table(labeled_df, workspace_root)
    colony_table, colony_cleaning_stats = _clean_colony_label_table(colony_table)
    colony_feature_cols = [
        c for c in colony_table.columns if c not in {"shape_label", "imaging_type", "image_id"}
    ]
```

In the `training_meta` dict (currently lines ~461-470), add:

```python
            "colony_rows_before_cleaning": int(colony_cleaning_stats["colony_rows_before_cleaning"]),
            "colony_rows_removed_by_cleaning": int(colony_cleaning_stats["colony_rows_removed_by_cleaning"]),
            "colony_cleaning_removals": dict(colony_cleaning_stats["colony_cleaning_removals"]),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_training_quality.py -q`
Expected: PASS (all, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/bacteria_assistant/training.py tests/test_training_quality.py
git commit -m "feat(training): record colony cleaning removals in training_meta"
```

---

### Task 3: image_id emission and image-level split for the shape model

**Files:**
- Modify: `src/bacteria_assistant/training.py` (`_build_colony_feature_table`, shape-model section of `train_models`)

**Interfaces:**
- Consumes: `colony_table["image_id"]` column
- Produces: `training._split_colonies_by_image(colony_table: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.Index, pd.Index]` (train indices, test indices)

- [ ] **Step 1: Write the failing tests**

In `tests/test_colony_cleaning.py`, append:

```python
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
```

Add `import numpy as np` to `tests/test_colony_cleaning.py` imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_colony_cleaning.py -q`
Expected: FAIL with `AttributeError: module 'bacteria_assistant.training' has no attribute '_split_colonies_by_image'`

- [ ] **Step 3: Emit image_id in _build_colony_feature_table**

In `src/bacteria_assistant/training.py`, inside `_build_colony_feature_table`'s colony loop, add after `features["shape_label"] = image_shape_label`:

```python
            features["image_id"] = str(image_path)
```

- [ ] **Step 4: Add _split_colonies_by_image**

In `src/bacteria_assistant/training.py`:
- Add `GroupShuffleSplit` to the `from sklearn.model_selection import ...` import.
- Add `import numpy as np` (check it is not already imported; if it is, skip).
- Add this function after `_clean_colony_label_table`:

```python
def _split_colonies_by_image(
    colony_table: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Split colony rows by image so no plate leaks between train and test (issue #6)."""
    if colony_table.empty or "image_id" not in colony_table.columns:
        raise ValueError("colony_table must be non-empty and contain an image_id column.")

    n_images = colony_table["image_id"].nunique()
    n_test_images = int(np.ceil(test_size * n_images))
    n_train_images = n_images - n_test_images
    if n_test_images == 0 or n_train_images == 0:
        raise ValueError(
            f"Cannot build an image-level split: {n_images} images yield "
            f"{n_train_images} train / {n_test_images} test images for test_size={test_size}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_pos, test_pos = next(splitter.split(colony_table, groups=colony_table["image_id"].to_numpy()))
    return colony_table.index[train_pos], colony_table.index[test_pos]
```

- [ ] **Step 5: Use the split for the shape model**

In `src/bacteria_assistant/training.py`, replace the shape-model block (currently lines ~429-438):

```python
    xs = colony_table[colony_feature_cols]
    ys = colony_table["shape_label"]

    xs_train, xs_test, ys_train, ys_test = train_test_split(
        xs,
        ys,
        test_size=0.2,
        random_state=random_state,
        stratify=ys,
    )

    shape_model, shape_summary, shape_model_name = _fit_best_ensemble_model(
        xs_train,
        ys_train,
        xs_test,
        ys_test,
        random_state=random_state,
        test_modalities=colony_table.loc[xs_test.index, "imaging_type"],
    )
```

with:

```python
    ys = colony_table["shape_label"]

    shape_train_idx, shape_test_idx = _split_colonies_by_image(
        colony_table,
        test_size=0.2,
        random_state=random_state,
    )
    xs_train = colony_table.loc[shape_train_idx, colony_feature_cols]
    ys_train = colony_table.loc[shape_train_idx, "shape_label"]
    xs_test = colony_table.loc[shape_test_idx, colony_feature_cols]
    ys_test = colony_table.loc[shape_test_idx, "shape_label"]

    shape_model, shape_summary, shape_model_name = _fit_best_ensemble_model(
        xs_train,
        ys_train,
        xs_test,
        ys_test,
        random_state=random_state,
        test_modalities=colony_table.loc[shape_test_idx, "imaging_type"],
    )
```

Important: the old `xs = colony_table[colony_feature_cols]` line must be deleted (it becomes
unused and ruff F841 flags unused locals). Keep `ys = colony_table["shape_label"]` — it feeds
`supported_shape_labels: sorted(set(ys))` later in `train_models`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_colony_cleaning.py tests/test_training_quality.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/bacteria_assistant/training.py tests/test_colony_cleaning.py
git commit -m "fix(training): prevent image-level leakage in shape split"
```

---

### Task 4: Local verification — lint, tests, retrain, evaluate

**Files:**
- Modify: `.gitignore` (add `dataset/`)

**Interfaces:**
- Consumes: the code from Tasks 1-3
- Produces: trained `artifacts/bacteria_models.joblib` + `artifacts/bacteria_models.metrics.json` with shape accuracy >= 0.72 and `training_meta` cleaning counts.

- [ ] **Step 1: Install ruff**

Run: `python -m pip install ruff==0.6.9`

- [ ] **Step 2: Run lint**

Run: `python -m ruff check .` then `python -m ruff format --check .`
Expected: clean (fix any violations, `python -m ruff format .` if needed).

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 4: Add dataset/ to .gitignore**

Append `dataset/` under the `# Data & Datasets` section of `.gitignore`.

- [ ] **Step 5: Commit housekeeping**

```bash
git add .gitignore
git commit -m "chore: ignore local dataset folder"
```

- [ ] **Step 6: Retrain**

Run:
```powershell
python scripts/train_model.py --dataset-csv dataset/dataset_full.csv --workspace-root D:\BPA_22
```
Expected: writes `artifacts/bacteria_models.joblib` and `artifacts/bacteria_models.metrics.json`; print output shows `colony_rows_before_cleaning`, `colony_rows_removed_by_cleaning`, `colony_cleaning_removals` in `training_meta`, and shape metrics.

- [ ] **Step 7: Evaluate**

Run: `python scripts/evaluate_models.py --metrics artifacts/bacteria_models.metrics.json`
Expected: `## shape_metrics` accuracy reported. Acceptance: accuracy >= 0.72 (baseline 0.675) and cocci/fungal recall measurably up vs baseline (cocci 0.35, fungal 0.13).

- [ ] **Step 8: Record evidence**

If accuracy >= 0.72, note the numbers for the PR. If it is below 0.72 but above baseline, tune `SHAPE_CLEANING_RULES` conservatively (repeat Tasks 1/4 cycle) and note the result.

- [ ] **Step 9: Upload artifact**

Run: `gh release upload model-artifact artifacts/bacteria_models.joblib artifacts/bacteria_models.metrics.json --clobber`
(Requires `gh` auth. The release already exists.)

---

### Task 5: PR, review, merge, closure note

**Files:** none (GitHub only)

- [ ] **Step 1: Push branch**

```bash
git push -u origin fix/issue-10-colony-labels
```

- [ ] **Step 2: Open PR**

```bash
gh pr create \
  --title "Fix #10: clean noisy colony labels + image-level shape split" \
  --body "## Linked issue
Closes #10, Closes #6

## What changed
- Drop colony rows whose geometry contradicts the parent shape_label (cocci/bacilli/fungal rules in config.py).
- Record cleaning removals in training_meta (before/removed/per-shape).
- Split colony shape train/test by image (GroupShuffleSplit) so no plate leaks across folds.

## Why
- Colony labels were inherited wholesale from the parent image, so the shape model learned label noise (accuracy 0.675).
- Same-image colonies straddling the split inflated accuracy via leakage.

## Validation
- [ ] Tests pass (pytest)
- [ ] Lint passes (ruff)
- [ ] Shape accuracy >= 0.72 (baseline 0.675) with cocci/fungal recall up; artifact reseeded

### Evidence
- Before: shape accuracy 0.675, cocci recall 0.35, fungal recall 0.13
- After: <FILL from make evaluate>

## Scope check
- [ ] This PR addresses only issue #10 (and its #6 dependency noted in the issue)
- [ ] Unrelated changes are excluded
"
```

- [ ] **Step 3: Verify CI**

Wait for lint/test to pass on the PR.

- [ ] **Step 4: Merge**

`gh pr merge <num> --squash --delete-branch` (or GitHub web "Squash and merge").

- [ ] **Step 5: Post closure note**

Comment on #10 with the resolved PR number, result numbers, validation summary, and any follow-ups.
