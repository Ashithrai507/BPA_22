# Issue #9 — Species Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise species-level accuracy from 0.341 toward/above 0.45 by adding shape/texture features, balancing imaging modalities within the training fold, and bumping the feature version.

**Architecture:** `features.py` gains three pure helpers (`_hu_log_scale`, `_colony_shape_features`, `_gradient_texture_features`) consumed by `extract_image_features`; `training.py` gains `_balance_modalities_per_species` applied in `train_models` between the image-level holdout split and augmentation, with new `training_meta` keys. All changes are unit-tested and verified by a local retrain + `evaluate_models.py`.

**Tech Stack:** Python 3.11, OpenCV (cv2), pandas, scikit-learn, numpy, pytest, ruff.

## Global Constraints

- Local env: Python 3.11.9, pytest 8.3.5, ruff 0.6.9 on PATH (Windows). No venv in repo.
- Run tests via `python -m pytest -q`; lint via `python -m ruff check .` and `python -m ruff format --check .`.
- Line length 120 (pyproject.toml ruff config). No new runtime dependencies.
- Dataset lives at `dataset/` (already ignored by `.gitignore` — do NOT touch it). Train/eval read CSV `dataset/dataset_full.csv`.
- `make train` equivalent: `python scripts/train_model.py --dataset-csv dataset/dataset_full.csv --workspace-root D:\BPA_22` (writes `artifacts/bacteria_models.joblib` + `.metrics.json`).
- Evaluate via `python scripts/evaluate_models.py --metrics artifacts/bacteria_models.metrics.json`.
- Baseline species (`organism_metrics`) accuracy: 0.341. Issue acceptance target: >= 0.45. **Best effort, honest report** — if under 0.45, keep the PR if accuracy improves and all gates pass, and file a follow-up.
- CI quality gate requires fungi recall >= 0.10 (keeps green after reseeding the artifact).
- Baseline test state (before this work): 24 passed, 2 skipped; ruff clean.
- Work starts from a NEW branch `fix/issue-9-species-features` off `origin/main`. Do NOT use the stale local `fix/issue-10-colony-labels`.

---

### Task 1: Create the feature branch and baseline checks

**Files:**
- Create (branch-level): none (git only)

**Interfaces:**
- Produces: branch `fix/issue-9-species-features` tracking `origin/main`; design + plan docs committed on it.

- [ ] **Step 1: Fetch and create the branch**

Run:
```powershell
git fetch origin
git checkout -b fix/issue-9-species-features origin/main
```
Expected: on branch `fix/issue-9-species-features`, working tree matches `origin/main`. Verify with `git log --oneline -1` → shows `a995f46 Merge pull request #13 ...`.

- [ ] **Step 2: Bring the design doc onto the branch**

The design spec `docs/superpowers/specs/2026-08-09-issue9-species-features-design.md` was committed on the old branch (`33010ac`). Cherry-pick it so the spec ships with the PR:
```powershell
git cherry-pick 33010ac
```
The plan doc `docs/superpowers/plans/2026-08-09-issue9-species-features.md` is untracked and carries over automatically with the checkout.

- [ ] **Step 3: Baseline verification**

Run:
```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```
Expected: `24 passed, 2 skipped`; ruff clean. This is the green baseline the rest of the plan must preserve.

- [ ] **Step 4: Commit plan doc**

```powershell
git add docs/superpowers/plans/2026-08-09-issue9-species-features.md
git commit -m "docs(plan): add issue #9 species features implementation plan"
```

---

### Task 2: Species feature helpers with unit tests

**Files:**
- Modify: `src/bacteria_assistant/features.py` (add helpers after `extract_image_features`; extend `extract_image_features`)
- Create: `tests/test_species_features.py`

**Interfaces:**
- Consumes: `_choose_best_mask(gray) -> list[np.ndarray]` (existing, in `features.py`)
- Produces:
  - `_hu_log_scale(hu: np.ndarray) -> np.ndarray` (elementwise `sign(x) * log1p(|x|)`)
  - `_colony_shape_features(gray: np.ndarray) -> dict[str, float]` with keys `hu_0..hu_6`, `colony_count`, `colony_hu_mean_0..colony_hu_mean_6`, `colony_hu_std_0..colony_hu_std_6`
  - `_gradient_texture_features(gray: np.ndarray) -> dict[str, float]` with keys `grad_mag_mean`, `grad_mag_std`, `grad_angle_std`
  - `extract_image_features` now returns those keys in addition to existing ones.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_species_features.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_species_features.py -q`
Expected: FAIL with `ImportError: cannot import name '_hu_log_scale'` (or `_colony_shape_features`).

- [ ] **Step 3: Add the helpers to features.py**

Add after the end of `extract_image_features` (before `def _threshold_mask`):

```python
def _hu_log_scale(hu: np.ndarray) -> np.ndarray:
    return np.sign(hu) * np.log1p(np.abs(hu))


def _colony_shape_features(gray: np.ndarray) -> dict[str, float]:
    contours = _choose_best_mask(gray)
    features: dict[str, float] = {"colony_count": float(len(contours))}
    for i in range(7):
        features[f"hu_{i}"] = 0.0
        features[f"colony_hu_mean_{i}"] = 0.0
        features[f"colony_hu_std_{i}"] = 0.0
    if not contours:
        return features

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    image_hu = _hu_log_scale(cv2.HuMoments(cv2.moments(mask)).flatten())
    for i, value in enumerate(image_hu):
        features[f"hu_{i}"] = float(value)

    per_colony = np.stack([_hu_log_scale(cv2.HuMoments(cv2.moments(c)).flatten()) for c in contours])
    mean = per_colony.mean(axis=0)
    std = per_colony.std(axis=0)
    for i in range(7):
        features[f"colony_hu_mean_{i}"] = float(mean[i])
        features[f"colony_hu_std_{i}"] = float(std[i])
    return features


def _gradient_texture_features(gray: np.ndarray) -> dict[str, float]:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    r = np.hypot(np.mean(np.cos(angle)), np.mean(np.sin(angle)))
    angle_std = float(np.sqrt(max(-2.0 * np.log(r + 1e-12), 0.0)))
    return {
        "grad_mag_mean": float(np.mean(mag)),
        "grad_mag_std": float(np.std(mag)),
        "grad_angle_std": angle_std,
    }
```

- [ ] **Step 4: Wire the helpers into extract_image_features**

In `extract_image_features`, immediately before `return features` (after the `gray_spatial_*` loop), add:

```python
    features.update(_colony_shape_features(gray))
    features.update(_gradient_texture_features(gray))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_species_features.py -q`
Expected: PASS (6 passed). Also re-run `python -m pytest tests/test_modality_fix.py -q` to confirm the existing feature tests still pass.

- [ ] **Step 6: Lint the changed files**

Run: `python -m ruff check src/bacteria_assistant/features.py tests/test_species_features.py`
Expected: clean. Auto-fix with `python -m ruff format src/bacteria_assistant/features.py tests/test_species_features.py` if needed.

- [ ] **Step 7: Commit**

```powershell
git add src/bacteria_assistant/features.py tests/test_species_features.py
git commit -m "feat(features): add Hu-moment and gradient texture features for species model"
```

---

### Task 3: Modality-balanced training fold with unit tests

**Files:**
- Modify: `src/bacteria_assistant/training.py` (add `_balance_modalities_per_species` after `_load_labeled_dataframe`)
- Create: `tests/test_species_features.py` (append balancing tests)

**Interfaces:**
- Consumes: nothing new (pure DataFrame function)
- Produces: `_balance_modalities_per_species(train_df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]`
  - Return 1: balanced DataFrame (species' over-represented modalities trimmed to the under-represented count; single-modality species untouched; empty input returned unchanged).
  - Return 2: `{species: {"before": {modality: count}, "after": {modality: count}, "balanced": bool}}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_species_features.py`:

```python
import pandas as pd

import bacteria_assistant.training as training


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
```

Note: `import pandas as pd` and `import bacteria_assistant.training as training` go near the top of the file with the other imports (pandas is a new import for this test file; training import is new).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_species_features.py -q`
Expected: FAIL with `AttributeError: module 'bacteria_assistant.training' has no attribute '_balance_modalities_per_species'`.

- [ ] **Step 3: Implement the balancing function**

In `src/bacteria_assistant/training.py`, add after `_load_labeled_dataframe` (before `_build_image_feature_table`):

```python
def _balance_modalities_per_species(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Trim over-represented imaging modalities per species within the train fold (issue #9).

    Each species with 2+ modalities is downsized so both modalities contribute the
    minimum observed count (no replacement, seeded). Single-modality species are
    untouched. Returns the balanced DataFrame and per-species audit stats.
    """
    if train_df.empty or "imaging_type" not in train_df.columns:
        return train_df.copy(), {}

    rng = np.random.default_rng(random_state)
    balanced_parts: list[pd.DataFrame] = []
    stats: dict[str, dict[str, Any]] = {}

    for species, group in train_df.groupby("organism", sort=True):
        modalities = [str(m) for m in sorted(group["imaging_type"].unique())]
        before = {m: int((group["imaging_type"] == m).sum()) for m in modalities}
        if len(modalities) < 2:
            balanced_parts.append(group)
            stats[str(species)] = {"before": before, "after": dict(before), "balanced": False}
            continue

        target = min(before.values())
        kept: list[pd.DataFrame] = []
        for modality in modalities:
            sub = group[group["imaging_type"] == modality]
            if len(sub) > target:
                positions = rng.choice(np.arange(len(sub)), size=target, replace=False)
                sub = sub.iloc[np.sort(positions)]
            kept.append(sub)
        balanced = pd.concat(kept)
        balanced_parts.append(balanced)
        after = {m: int((balanced["imaging_type"] == m).sum()) for m in modalities}
        stats[str(species)] = {"before": before, "after": after, "balanced": True}

    result = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else train_df.copy()
    return result, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_species_features.py -q`
Expected: PASS (12 passed — 6 feature + 6 balancing).

- [ ] **Step 5: Lint and commit**

Run: `python -m ruff check src/bacteria_assistant/training.py tests/test_species_features.py` (auto-format if needed).

```powershell
git add src/bacteria_assistant/training.py tests/test_species_features.py
git commit -m "feat(training): balance imaging modalities per species in train fold"
```

---

### Task 4: Wire into train_models, bump FEATURE_VERSION, add integration test

**Files:**
- Modify: `src/bacteria_assistant/config.py` (`FEATURE_VERSION` 2 → 3)
- Modify: `src/bacteria_assistant/training.py` (`train_models` split/table section; `training_meta` dict)
- Test: `tests/test_species_features.py` (append integration test)

**Interfaces:**
- Consumes: `_balance_modalities_per_species(train_df, random_state) -> (df, stats)` (Task 3)
- Produces: `training_meta` keys `train_image_count_before_balancing`, `train_image_count_after_balancing`, `modality_balance_stats`, `image_feature_count`; artifact `feature_version = 3`.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_species_features.py`:

```python
from sklearn.dummy import DummyClassifier


def test_train_models_records_modality_balance_meta(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "organism": ["Bacillus subtilis"] * 4 + ["Staphylococcus aureus"] * 16,
            "image_path": [f"img_{i}.png" for i in range(20)],
            "imaging_type": (
                ["gram stain"] * 2
                + ["media plate"] * 2
                + ["gram stain"] * 8
                + ["media plate"] * 8
            ),
        }
    ).to_csv(csv_path, index=False)

    captured: dict[str, pd.DataFrame] = {}

    def fake_split(df, test_size=None, random_state=None, stratify=None):
        n_train = int(len(df) * 0.8)
        return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()

    def fake_image_table(labeled_df, workspace_root, augment=False) -> pd.DataFrame:
        if augment:
            captured["train"] = labeled_df.copy()
        return pd.DataFrame(
            {
                "image_path": ["img_0.png", "img_1.png"],
                "organism": ["Bacillus subtilis", "Staphylococcus aureus"],
                "organism_type": ["bacteria", "bacteria"],
                "gram_label": ["gram_positive", "gram_positive"],
                "shape_label": ["bacilli", "cocci"],
                "taxonomy_group": ["gram_positive_bacilli", "gram_positive_cocci"],
                "imaging_type": ["gram stain", "gram stain"],
                "r_mean": [0.5, 0.6],
            }
        )

    monkeypatch.setattr(training, "train_test_split", fake_split)
    monkeypatch.setattr(training, "_build_image_feature_table", fake_image_table)
    monkeypatch.setattr(
        training,
        "_build_colony_feature_table",
        lambda labeled_df, workspace_root: pd.DataFrame(
            {
                "image_id": ["img_a", "img_b", "img_c", "img_d", "img_e", "img_f"],
                "shape_label": ["cocci", "cocci", "bacilli", "bacilli", "fungal", "spiral"],
                "aspect_ratio": [1.1, 3.0, 1.8, 1.05, 1.4, 2.5],
                "solidity": [0.97, 0.98, 0.96, 0.93, 0.99, 0.9],
                "imaging_type": ["gram", "gram", "plate", "plate", "gram", "plate"],
            }
        ),
    )
    monkeypatch.setattr(
        training,
        "_fit_best_ensemble_model",
        lambda x_train, y_train, *a, **k: (
            DummyClassifier(strategy="most_frequent").fit(x_train, y_train),
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
    assert meta["train_image_count_before_balancing"] == 16
    assert meta["train_image_count_after_balancing"] == 12
    assert meta["image_feature_count"] == 1
    assert meta["modality_balance_stats"]["Staphylococcus aureus"]["after"] == {
        "gram stain": 4,
        "media plate": 4,
    }
    aureus = captured["train"][captured["train"]["organism"] == "Staphylococcus aureus"]
    assert aureus["imaging_type"].value_counts().to_dict() == {"gram stain": 4, "media plate": 4}
```

Note: add `import pandas as pd` (already added in Task 3), `import bacteria_assistant.training as training` (already added), and `from sklearn.dummy import DummyClassifier` to the imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_species_features.py::test_train_models_records_modality_balance_meta -q`
Expected: FAIL with `KeyError: 'train_image_count_before_balancing'`.

- [ ] **Step 3: Bump FEATURE_VERSION**

In `src/bacteria_assistant/config.py`, change:
```python
FEATURE_VERSION = 2
```
to:
```python
FEATURE_VERSION = 3
```

- [ ] **Step 4: Apply balancing in train_models**

In `src/bacteria_assistant/training.py`, replace:
```python
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=random_state,
        stratify=labeled_df["organism"],
    )

    train_table = _build_image_feature_table(train_df, workspace_root, augment=True)
```
with:
```python
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=random_state,
        stratify=labeled_df["organism"],
    )
    balanced_train_df, modality_balance_stats = _balance_modalities_per_species(train_df, random_state)

    train_table = _build_image_feature_table(balanced_train_df, workspace_root, augment=True)
```

- [ ] **Step 5: Record new training_meta keys**

In the `training_meta` dict inside `train_models`, right after the existing `"train_image_count": int(len(train_df)),` line, add:
```python
            "train_image_count_before_balancing": int(len(train_df)),
            "train_image_count_after_balancing": int(len(balanced_train_df)),
            "modality_balance_stats": modality_balance_stats,
```
And after the existing `"feature_version": FEATURE_VERSION,` line inside `training_meta`, add:
```python
            "image_feature_count": int(len(image_feature_cols)),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_species_features.py -q`
Expected: PASS (13 passed). Then run the full suite:
Run: `python -m pytest -q`
Expected: 37 passed, 2 skipped (24 baseline + 13 new; the baseline 2 skips remain).

- [ ] **Step 7: Lint and commit**

Run: `python -m ruff check .` and `python -m ruff format --check .` (auto-format if needed).

```powershell
git add src/bacteria_assistant/config.py src/bacteria_assistant/training.py tests/test_species_features.py
git commit -m "feat(training): bump FEATURE_VERSION to 3 and record modality balance meta"
```

---

### Task 5: Local verification — lint, tests, retrain, evaluate

**Files:**
- Modify: `artifacts/bacteria_models.joblib`, `artifacts/bacteria_models.metrics.json` (regenerated)

**Interfaces:**
- Consumes: code from Tasks 2-4
- Produces: retrained artifact with `feature_version: 3`, new feature columns, and `training_meta` balance keys; evidence numbers for the PR.

- [ ] **Step 1: Lint**

Run: `python -m ruff check .` then `python -m ruff format --check .`
Expected: clean.

- [ ] **Step 2: Full test suite**

Run: `python -m pytest -q`
Expected: all pass (2 pre-existing skips allowed).

- [ ] **Step 3: Retrain**

Run:
```powershell
python scripts/train_model.py --dataset-csv dataset/dataset_full.csv --workspace-root D:\BPA_22
```
This takes a while (~629 images × 4 variants, colony extraction per image). Expected: writes `artifacts/bacteria_models.joblib` + `.metrics.json`; output JSON contains `"feature_version": 3`, `"image_feature_count": <~640>`, `"train_image_count_before_balancing"`, `"train_image_count_after_balancing"`, and `"modality_balance_stats"`.

- [ ] **Step 4: Evaluate and record before/after**

Run: `python scripts/evaluate_models.py --metrics artifacts/bacteria_models.metrics.json`
Record for the PR evidence:
- `organism_metrics` accuracy (baseline 0.341; target >= 0.45).
- Per-group `group_species_metrics[...]` accuracies (baseline: fungi 0.72, gram_positive_bacilli 0.61, gram_positive_cocci 0.55, gram_negative_bacilli 0.36).
- `organism_type_metrics` fungi recall (CI gate needs >= 0.10).

Save the numbers into a scratch note file `C:\Users\chira\AppData\Local\Temp\opencode\issue9-evidence.txt` for use in the PR body.

- [ ] **Step 5: Decision point**

- If `organism_metrics` accuracy >= 0.45: acceptance met, continue.
- If below 0.45 but above baseline 0.341: per the agreed "best effort, honest report" policy, keep the PR, report the achieved value honestly, and open a follow-up issue noting the augmentation-bump (Approach C) path. Continue to Task 6.
- If accuracy regressed below 0.341: stop, debug, and reconsider before proceeding.

- [ ] **Step 6: Commit the retrained artifact**

```powershell
git add artifacts/bacteria_models.joblib artifacts/bacteria_models.metrics.json
git commit -m "chore(models): retrain artifact with issue-9 species features"
```

- [ ] **Step 7: Reseed the artifact release for CI**

Run: `gh release upload model-artifact artifacts/bacteria_models.joblib artifacts/bacteria_models.metrics.json --clobber`
(Requires `gh` auth; the `model-artifact` release already exists.)

---

### Task 6: Push, PR, CI, merge, closure note

**Files:** none (GitHub only)

- [ ] **Step 1: Push branch**

```powershell
git push -u origin fix/issue-9-species-features
```

- [ ] **Step 2: Open PR**

```powershell
gh pr create \
  --title "Fix #9: species accuracy via Hu-moment features + modality-balanced training" \
  --body "## Linked issue
Closes #9

## What changed
- Add image-mask Hu moments (7), per-colony Hu moment aggregates (mean/std, 14), colony_count, and gradient texture stats (3) to \`extract_image_features\`.
- Add \`_balance_modalities_per_species\`: trims over-represented imaging modality per species in the train fold so modality cannot be a species shortcut.
- Bump \`FEATURE_VERSION\` to 3; record \`train_image_count_before/after_balancing\`, \`modality_balance_stats\`, \`image_feature_count\` in \`training_meta\`.

## Why
- Species accuracy was near-random (0.341); per-modality gap (gram 0.43 / media 0.24) showed a modality shortcut.

## Validation
- [ ] Tests pass (pytest)
- [ ] Lint passes (ruff)
- [ ] Metrics/artifacts updated (FEATURE_VERSION 3, artifact reseeded)

### Evidence
- Before: organism accuracy 0.341; groups fungi 0.72 / gp_bacilli 0.61 / gp_cocci 0.55 / gn_bacilli 0.36
- After: <FILL from Task 5 evidence>

## Scope check
- [ ] This PR addresses only issue #9
- [ ] Unrelated changes are excluded
"
```
Replace `<FILL from Task 5 evidence>` with the recorded numbers before submitting.

- [ ] **Step 3: Verify CI**

Wait for lint/test/model-quality gates to pass on the PR. If the quality gate fails on fungi recall, re-check the evidence and report in the PR.

- [ ] **Step 4: Squash-merge to main**

`gh pr merge <num> --squash --delete-branch` (or GitHub web "Squash and merge"). Confirm the merged commit lands on `main`.

- [ ] **Step 5: Post closure note on #9**

```markdown
Resolved in PR #<pr_number>.

### Result
- organism accuracy: <achieved> (baseline 0.341)
- per-group specialists: <values>

### Validation summary
- Tests: <n> passed
- Metrics/artifacts: FEATURE_VERSION 3, artifact reseeded
- Follow-ups (if any): #<new_issue_number>
```
