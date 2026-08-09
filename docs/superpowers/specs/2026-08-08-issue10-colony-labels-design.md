# Design: Noisy colony ground-truth limits shape model (issue #10) + image-level split (issue #6)

Date: 2026-08-08

## Summary

The colony shape model is trained on noisy labels: every colony row inherits its parent
image's `shape_label` wholesale (`src/bacteria_assistant/training.py:86-89`), even though a
single plate can contain mixed morphology. The model therefore learns label noise
(shape accuracy 0.675, fungal recall 0.13, cocci recall 0.35). Additionally, colonies from
the same image can straddle the train/test split, inflating measured accuracy via leakage.

This design fixes both, in a single PR:

- #10: heuristic cleaning of colony rows whose geometry strongly contradicts the parent label.
- #6:  image-level train/test split so no plate leaks between folds.

## Approach

### 1. Cleaning filter (issue #10)

Add a pure, unit-testable function in `training.py`:

```python
def _clean_colony_label_table(
    colony_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
```

- Input: the raw colony feature table (one row per colony, with `shape_label`).
- Applies `SHAPE_CLEANING_RULES` from `config.py`.
- Returns the filtered frame and a stats dict for auditability:
  - `colony_rows_before_cleaning`
  - `colony_rows_removed_by_cleaning`
  - `colony_cleaning_removals` (per-shape counts)

Rules (conservative start) live in `config.py`:

| shape_label | condition to drop row |
|-------------|-----------------------|
| `cocci`     | `aspect_ratio > 2.0` |
| `bacilli`   | `aspect_ratio < 1.2` |
| `fungal`    | `solidity > 0.95`    |
| `spiral`    | none (features cannot measure spiral morphology) |

`train_models` calls `_clean_colony_label_table` immediately after
`_build_colony_feature_table`, and records the stats in `training_meta`.

### 2. Image-level split (issue #6)

- `_build_colony_feature_table` also emits an `image_id` column (stable per source image).
- `image_id` is excluded from `colony_feature_cols`.
- Replace `train_test_split` at `training.py:432-438` with a new helper:

```python
def _split_colonies_by_image(
    colony_table: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
```

- Uses sklearn `GroupShuffleSplit` on `image_id` (n_splits=1).
- Guarantees zero shared `image_id` between train and test index sets.
- Raises `ValueError` with a clear message if the split yields an unusable test set
  (no test samples, or too few groups).

### 3. Tests (`tests/test_colony_cleaning.py`)

- Cleaning filter removes elongated-cocci, smooth-fungal, and round-bacilli rows; keeps
  valid rows; never removes `spiral`.
- Removal counts reported per shape and totals are correct.
- Split yields zero shared `image_id` between train/test.
- Split is reproducible for the same seed.
- Cleaning is a no-op when no row contradicts its label.

### 4. Verification

- Local: `make test` and `make lint` (no dataset required).
- Accuracy verification requires the dataset (not available in this workspace): the
  maintainer runs `make train` + `make evaluate` and confirms shape accuracy >= 0.72
  (baseline 0.675) and measurably higher cocci/fungal recall, then runs
  `make ci-upload-artifact` before merge.

### 5. Branch

`fix/issue-10-colony-labels` off `main` (independent of the still-open issue #7 PR).

## Acceptance criteria

- Shape model accuracy >= 0.72 (baseline 0.675); cocci/fungal recall measurably up.
- Removal counts recorded in `training_meta`.
- Zero `image_id` overlap between shape train/test folds.
- Existing tests + lint green; unit tests cover the cleaning filter and the split.

## Out of scope

- Fixing fungi recall in the organism-type model (issue #7).
- Other morphology signals (circularity, intensity) in the cleaning rules — can be added
  later if accuracy still lags.
- Changing the inference path; the trained shape model interface is unchanged.
