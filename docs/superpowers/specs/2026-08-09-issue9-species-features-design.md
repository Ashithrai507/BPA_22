# Issue #9 — Species accuracy near-random at 0.333 (Phase 3 - A3)

Date: 2026-08-09

## Problem

Species-level classification is near-random. On the current artifact (`artifacts/bacteria_models.metrics.json`):

- `organism_metrics` accuracy 0.341 (baseline in the issue: ~0.333; 10 classes → 0.10 by chance).
- Per-modality species accuracy: gram stain 0.433, media plate 0.237 — a large gap that indicates the
  model partially uses the imaging modality as a species shortcut (issue #8 confounder).
- Only *Staphylococcus aureus* and *Bacillus subtilis* are meaningfully learned at the global level;
  *Klebsiella pneumoniae*, *Enterococcus faecalis*, *Clostridium sporogenes* are coin flips.
- Per-group specialists: fungi 0.72, gram_positive_bacilli 0.61, gram_positive_cocci 0.55,
  gram_negative_bacilli 0.36.

## Scope (Phase 3 of the accuracy-improvement plan)

Only issue #9. Issue #8 (CLAHE + photometric augmentation) and #7 (fungi recall) are already merged to
`main`; this work builds on top of them. No changes to the colony shape model or inference contract.

## Approach

1. **Modality-balanced training fold** — for each species, trim the over-represented modality down to
   the under-represented one so modality cannot be used as a species shortcut. Applied to the train
   fold only (after the image-level holdout split, before augmentation); the test fold stays in
   natural distribution.
2. **Photometric augmentation** — already present from #8; reused unchanged.
3. **Stronger features (no new deps)** in `src/bacteria_assistant/features.py`:
   - Image-mask Hu moments (7 dims): union mask of all valid colony contours → `cv2.HuMoments`,
     log-scaled (`sign(x) * log1p(|x|)`); zeros on empty mask.
   - Colony aggregate features (15 dims): per-colony Hu moments (7) aggregated as mean/std across
     colonies, plus `colony_count`. Computed inside `extract_image_features` from the raw contours;
     `ColonyMeasurement` / `colony_to_feature_dict` / shape model are untouched.
   - Gradient texture stats (3 dims): Sobel gradient magnitude mean/std and circular std of the
     orientation angle, computed on the CLAHE gray image.
4. **Retrain & reseed** — bump `FEATURE_VERSION` to 3, retrain, `make evaluate`, upload artifact.

## Architecture

- `features.py`: new pure helpers operating on the CLAHE gray image and contour lists, consumed by
  `extract_image_features` (already called per image and per augmented variant).
- `training.py`: new `_balance_modalities_per_species(train_df, random_state) -> (df, stats)` applied
  in `train_models` between `train_test_split` and `_build_image_feature_table(augment=True)`.
- `config.py`: `FEATURE_VERSION = 3`.
- `training_meta` gains: `train_image_count_before_balancing`, `train_image_count_after_balancing`,
  `modality_balance_stats` (per-species modality counts), `image_feature_count`.
- New feature columns flow automatically into `image_feature_cols` (derived by exclusion); the full
  list is stored in the joblib artifact's `image_feature_columns`.

## Error handling

- Empty colony mask / no colonies → Hu moments zeros, `colony_count = 0` (no NaN leakage into models).
- Species with a single modality → left untouched by balancing; species never collapses to zero rows.
- `grad_angle_std` guarded with a small constant against division by zero.

## Testing

New `tests/test_species_features.py`:

- `extract_image_features` on a synthetic image returns the new keys and is deterministic.
- `_balance_modalities_per_species` equalizes imbalanced species, leaves single-modality species
  untouched, is reproducible, does not mutate input, and never empties a species.
- `train_models` records the new `training_meta` keys (monkeypatched table builders, mirroring the
  existing colony-cleaning integration test).
- `FEATURE_VERSION == 3`.

## Acceptance criteria

- Species accuracy (`organism_metrics`) >= 0.45 on the artifact (baseline 0.341). **Best effort,
  honest report**: if 0.45 is not reached, report the achieved value, keep the PR if accuracy
  improves and all gates pass, and file a follow-up issue (augmentation bump as the next lever).
- Group-specialist models improve or hold (measured per group in PR evidence).
- New feature columns present in the artifact's `image_feature_columns`; `image_feature_count`
  recorded in the metrics JSON's `training_meta`.
- Lint (`ruff check` + `ruff format --check`) and full test suite green.
- CI quality gate (fungi recall >= 0.10) stays green after reseeding `model-artifact`.

## Delivery

- New branch `fix/issue-9-species-features` off `origin/main` (current local branch
  `fix/issue-10-colony-labels` is a stale duplicate of already-merged work and is not used).
- Commits: tests → features → balancing → versioning/meta → retrain/artifact.
- Push, open PR with `Closes #9` using the issue's PR template, wait for CI, squash-merge to `main`,
  post closure note on #9.
