# Design: Integrate DL embedding training and inference from BPA22_clone

Date: 2026-08-13

## Summary

The main repo (`BPA_22`) previously only shipped classical ML. The deep-learning
integration — a DL embedding model (`efficientnet_b0` backbone), a `train_dl`
CLI, an embedding-based retrain path for the sklearn hierarchy, and
embedding-aware inference (DL species head + TTA, group-constrained) — was
developed in the `BPA22_clone` workspace. The `src/bacteria_assistant/dl/`
package, `_build_image_embedding_table`, `FEATURE_VERSION=3`, torch deps, and
pyproject/ruff entries were already ported over (uncommitted).

This design finishes the port: wire the embedding path into main's `train_models`
and `predict_bacteria_image`, add `scripts/train_dl.py` and the `--dl` flag on
`scripts/train_model.py`, port the DL tests, install torch in the venv, and
commit to `main`.

The main repo has since evolved past the clone (photometric augmentation,
colony-label cleaning, shared image-level holdout, image-quality rejection).
The port must merge into that newer architecture, not overwrite it.

## Approach

### 1. `src/bacteria_assistant/training.py` — embedding path in `train_models`

Add `embedding_model_path: str | Path | None = None` to `train_models(...)`.

When provided:

- Load the checkpoint: `dl.extract.load_embedding_model(path)` → model, encoders, config.
- Read `embedding_dim` and `input_size` from the checkpoint config.
- Build `train_table = _build_image_embedding_table(train_df, ...)` and
  `test_table = _build_image_embedding_table(test_df, ...)` from the **already-split**
  folds, so the shared holdout stays leakage-free (the helper's docstring already
  assumes caller-provided splits).
- `image_feature_cols` = the `emb_*` columns.
- Stash `artifacts["embedding_model_path"] = str(Path(path).name)`.

When absent: current behavior unchanged (classical features, `augment=True` on
train fold). All downstream model fitting (gram / organism_type / group /
organism / group_species / shape) is feature-agnostic and unchanged.

### 2. `scripts/train_model.py` — `--dl` flag

Mirror the clone: add `--dl PATH` argument, pass through as
`embedding_model_path=args.dl`.

### 3. `scripts/train_dl.py` — new CLI

Port from the clone, adapted to main's layout:

- `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (scripts/ layout, matching
  `scripts/train_model.py`).
- Default `--dataset-csv` = `PROJECT_ROOT / DATASET_ROOT / "dataset_full.csv"`.
- Default `--output` = `PROJECT_ROOT / "artifacts" / "embedding_model.pt"`.
- Reuse `_load_labeled_dataframe` and the existing `dl.*` API unchanged.

### 4. `src/bacteria_assistant/inference.py` — embedding-aware inference

- `load_models()`: after `joblib.load`, call a new `_attach_embedding_model(artifacts, artifacts_dir)`
  which loads the DL checkpoint when `artifacts["embedding_model_path"]` is present and
  resolvable, attaching `_embedding_model`, `_embedding_encoders`,
  `_embedding_config`, `_embedding_input_size`. No-op when absent (classical artifacts
  keep the old path).
- `predict_bacteria_image()`:
  - Always compute `image_features` (keeps the `_should_reject_image` quality gate
    using `edge_density` / `laplacian_var` working).
  - If the DL model is attached, build the feature vector from `emb_*` columns instead
    of classical features.
  - Species prediction: when a DL species encoder is present, use
    `dl.extract.species_probabilities(...)` with `tta=True`, restricted to
    `ORGANISMS_BY_GROUP[predicted_group]` (fallback to argmax over all species).
    Otherwise use the existing `_predict_species_with_group_constraint` (now running
    on embeddings).
  - Keep the `_should_reject_image` gate and `UNKNOWN_LABEL` response unchanged.

### 5. Tests — port the DL suite

Port all 7 files from the clone: `test_dl_dataset.py`, `test_dl_extract.py`,
`test_dl_inference.py`, `test_dl_losses.py`, `test_dl_train_sklearn.py`,
`test_dl_training.py`, `test_embedding_model.py`.

Only adaptation needed: `test_dl_inference._first_image()` should use main's
`DATASET_ROOT` → `Bacteria dataset` fallback convention (as `test_output_contract.py`).
The remaining tests are self-contained (tmp_path fixtures, `TinyBackbone`,
`pretrained=False` models).

### 6. Environment

Install `torch==2.13.0` and `torchvision==0.28.0` into `BPA_22/.venv` (exact versions
already in the clone's venv and already listed in `requirements.txt`).

### 7. Verification

- `make test` (full suite incl. DL tests) green.
- `make lint` green.

### 8. Commit

Single conventional commit on `main`:
`feat: port DL embedding training and inference from BPA22_clone`

## Acceptance criteria

- `train_models(..., embedding_model_path=...)` trains the sklearn hierarchy on DL
  embeddings with a leakage-free shared holdout; classical path unchanged.
- `predict_bacteria_image` uses the DL species head (TTA + group constraint) for
  DL artifacts and falls back to classical for non-DL artifacts; quality gate intact.
- `scripts/train_dl.py` and `scripts/train_model.py --dl` run.
- All 7 DL tests + existing suite + lint pass in `BPA_22/.venv`.

## Out of scope

- Retraining/regenerating the trained artifact `artifacts/bacteria_models.joblib`
  (requires the dataset + GPU; a follow-up `make train --dl` step).
- Porting clone-only docs (`docs/step_log.md`, `documentation/developer.md`,
  `documentation/image_processing_pipeline.md`).
- The clone's cosmetic-only `bacteria_ui.py` import reordering.
