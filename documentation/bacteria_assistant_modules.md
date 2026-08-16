# `bacteria_assistant/` — Module Reference (Phase 1: CV/ML Pipeline)

> A complete, function-by-function explanation of every module in
> `src/bacteria_assistant/` — the classical ML pipeline plus the deep-learning
> embedding sub-module. Read this together with
> [`developer.md`](developer.md) (end-to-end walkthrough) and
> [`image_processing_pipeline.md`](image_processing_pipeline.md) (image/CV detail).

---

## 0. The mental model

```
                     src/bacteria_assistant/
        ┌────────────────────────────────────────────────────────────┐
        │ config.py        ← the taxonomy + threshold "constitution" │
        │ features.py      ← hand-crafted CV: 616-dim image, 7-dim   │
        │                    colony features, CLAHE, augmentation,    │
        │                    colony detection & distribution          │
        │ training.py      ← hierarchical sklearn classifier stack    │
        │ inference.py     ← prediction orchestration + JSON output   │
        │ dl/              ← PyTorch embedding model + training +     │
        │                    inference-time extraction (replaces the  │
        │                    616-dim image features at inference)     │
        └────────────────────────────────────────────────────────────┘
```

Two "worlds" of features can feed the sklearn hierarchy:

- **Classical (baseline / fallback):** 616-dim vector from `features.extract_image_features`.
- **Deep-learning (primary):** 1280-dim embedding from `dl/extract.extract_embedding`.

`inference.py` chooses automatically based on whether the artifact bundle points to a
`.pt` checkpoint (`embedding_model_path`). Old artifacts keep working unchanged.

---

## 1. `config.py` — the taxonomy contract and all thresholds

This file defines **everything the rest of the package treats as ground truth**.
`FEATURE_VERSION` is the single most important knob: bump it whenever feature
extraction changes so stale artifacts are forced to be retrained.

### 1.1 `ORGANISM_METADATA`

A fixed dict mapping each of the **10 supported organisms** to its biological labels:

```python
"Staphylococcus aureus": {
    "organism_type": "bacteria",
    "gram_label": "gram_positive",
    "shape_label": "cocci",
    "taxonomy_group": "gram_positive_cocci",
}
```

All four label columns in the training DataFrames are **derived** from this dict,
never stored. This guarantees label consistency — the taxonomy lives in exactly one
place. Inference also uses it for the **metadata consistency override**: the biology
table always wins over the classifiers (see §4.5).

| Organism | organism_type | gram_label | shape_label | taxonomy_group |
| -------- | ------------- | ---------- | ----------- | -------------- |
| Staphylococcus aureus | bacteria | gram_positive | cocci | gram_positive_cocci |
| Streptococcus pyogenes | bacteria | gram_positive | cocci | gram_positive_cocci |
| Enterococcus faecalis | bacteria | gram_positive | cocci | gram_positive_cocci |
| Bacillus subtilis | bacteria | gram_positive | bacilli | gram_positive_bacilli |
| Clostridium sporogenes | bacteria | gram_positive | bacilli | gram_positive_bacilli |
| Escherichia coli | bacteria | gram_negative | bacilli | gram_negative_bacilli |
| Klebsiella pneumoniae | bacteria | gram_negative | bacilli | gram_negative_bacilli |
| Pseudomonas aeruginosa | bacteria | gram_negative | bacilli | gram_negative_bacilli |
| Candida albicans | fungi | non_bacterial_fungi | fungal | fungi |
| Aspergillus niger | fungi | non_bacterial_fungi | fungal | fungi |

### 1.2 `SUPPORTED_SHAPES` / `UNKNOWN_LABEL`

- `SUPPORTED_SHAPES = ("cocci", "bacilli", "spiral", "fungal")` — the shape label
  vocabulary. Note `spiral` has **no training samples**; it is inferred at inference
  time by a geometry heuristic (§4.6).
- `UNKNOWN_LABEL = "unknown"` — the value emitted for every label field when the
  image is rejected by the confidence gates.

### 1.3 `SHAPE_CLEANING_RULES` (issue #10)

Heuristic label-cleaning for the colony shape model. A colony row is **dropped** from
training when its geometry strongly contradicts its parent image's `shape_label`:

```python
SHAPE_CLEANING_RULES = {
    "cocci":   {"max_aspect_ratio": 2.0},   # a "cocci" colony can't be a long rod
    "bacilli": {"min_aspect_ratio": 1.2},   # a "bacilli" colony must be elongated
    "fungal":  {"max_solidity": 0.95},      # fungal growth is irregular, not convex
}
```

### 1.4 Rejection thresholds (confidence gating)

| Constant | Value | Meaning |
| -------- | ----- | ------- |
| `MIN_ORGANISM_CONFIDENCE` | 0.55 | species/organism confidence floor |
| `MIN_COLONY_CONFIDENCE` | 0.45 | shape confidence floor |
| `MIN_COLONIES_FOR_VALID` | 1 | at least one colony must exist |
| `MIN_ORGANISM_TYPE_PROB` | 0.6 | organism_type probability floor |
| `MIN_EDGE_DENSITY` | 0.015 | image sharpness gate (edges) |
| `MIN_LAPLACIAN_VAR` | 25.0 | image sharpness gate (focus) |

### 1.5 `ORGANISMS_BY_GROUP`

The species-per-group mapping used for **group-constrained species prediction**. This
is the mechanism that stops biologically impossible outputs (e.g. a coccus predicted
as a gram-negative bacillus):

```python
"gram_positive_cocci":   ["Staphylococcus aureus", "Streptococcus pyogenes", "Enterococcus faecalis"],
"gram_positive_bacilli": ["Bacillus subtilis", "Clostridium sporogenes"],
"gram_negative_bacilli": ["Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa"],
"fungi":                 ["Candida albicans", "Aspergillus niger"],
```

### 1.6 Paths, feature version, CLAHE, augmentation

- **Paths** are overridable via env vars: `BACTERIA_DATASET_ROOT` → `data/dataset`,
  `BACTERIA_ARTIFACT_DIR` → `artifacts`, `BACTERIA_MODEL_PATH` → `artifacts/bacteria_models.joblib`.
- `FEATURE_VERSION = 3` — bumped whenever feature extraction changes; stored in the
  artifact's `training_meta` and metrics so you can detect stale artifacts.
- `CLAHE_CLIP_LIMIT = 2.0`, `CLAHE_TILE_GRID = 8` — illumination normalization
  (removes the gram-stain vs media-plate confounder, issue #8).
- Augmentation knobs: `AUGMENT_PER_IMAGE = 3` variants, brightness σ = 12, contrast
  α ∈ (0.85, 1.15), CLAHE clip ∈ (1.0, 3.0). Applied to the **training fold only**.

### 1.7 `normalize_organism_name(name)`

Normalizes species strings (`"Bacillus_subtilis"` → `"Bacillus subtilis"`) so CSV
labels always match `ORGANISM_METADATA` keys regardless of underscore/spacing quirks.

---

## 2. `features.py` — feature extraction + colony computer vision

### 2.1 Image I/O

- `read_image(image_path)` — reads via `np.fromfile` + `cv2.imdecode` (not
  `cv2.imread`) so paths with spaces/unicode work. Raises `FileNotFoundError` on failure.
- `augment_image(image, seed)` — produces `AUGMENT_PER_IMAGE` photometric variants:
  contrast `×α` + brightness `+β`, then CLAHE on the L channel of LAB. Deterministic
  per seed (used with an md5-of-path seed in training).

### 2.2 `extract_image_features(image)` → 616-dim dict

Called by classical training and by `predict_bacteria_image` for the quality gates.
The image is resized to **256×256** first. Feature groups:

| Group | Count | Details |
| ----- | ----- | ------- |
| RGB stats | 6 | r/g/b mean + std |
| HSV stats | 6 | h/s/v mean + std |
| gray stats | 2 | gray_mean, gray_std (gray is **CLAHE-normalized**) |
| texture | 2 | `laplacian_var` (focus), `edge_density` (Canny 70/160) |
| RGB histograms | 24 | 8-bin per-channel normalized histograms |
| gray spatial signature | 576 | 24×24 downsampled grayscale grid (`/255.0`) — encodes plate layout |
| colony shape (Hu) | 22 | see `_colony_shape_features` |
| gradient texture | 3 | Sobel magnitude mean/std + angle std |

**Key design points:**
- The **spatial signature (576 dims)** was the single biggest classical accuracy boost.
- CLAHE is applied **before** edge/Laplacian/spatial features so lighting modality
  doesn't dominate.

#### `_colony_shape_features(gray)` — Hu moments (22 features)

Runs the same segmentation as colony detection (§2.4), then computes:
- `colony_count`
- `hu_0..hu_6` — log-scaled Hu moments of the whole mask (7)
- `colony_hu_mean_0..6`, `colony_hu_std_0..6` — mean/std of per-contour Hu moments (14)

`_hu_log_scale` applies `sign(x)·log1p(|x|)` so moments don't explode in magnitude.

#### `_gradient_texture_features(gray)` — 3 features

Sobel gradient magnitude mean/std + circular angle dispersion
(`angle_std = sqrt(max(-2·log(r), 0))` where `r` is the mean resultant length).

### 2.3 Colony segmentation (the heuristic segmenter)

This is classical CV, no ML. Pipeline:

1. Grayscale + CLAHE.
2. `_threshold_mask(gray, invert)` — GaussianBlur (5×5) → **Otsu** threshold
   (plain or inverted) → morphological **open + close** (3×3).
3. `_valid_contours(mask, image_area)` — `findContours(RETR_EXTERNAL,
   CHAIN_APPROX_SIMPLE)`, keep contours with
   `max(20, image_area·0.00003) ≤ area ≤ image_area·0.12`.
4. `_choose_best_mask(gray)` — run both inverted and plain masks and **pick the one
   with the richer contour set** (tie → inverted). Handles either being empty.

### 2.4 `extract_colonies(image)` → list[`ColonyMeasurement`]

The public entry point: CLAHE gray → `_choose_best_mask` → one `ColonyMeasurement`
per contour, **sorted by area descending**.

`ColonyMeasurement` (dataclass) holds the 7 shape features + centroid:

| Field | Formula |
| ----- | ------- |
| `area` | `cv2.contourArea` |
| `perimeter` | `cv2.arcLength(..., closed=True)` |
| `circularity` | `4πA / P²` |
| `aspect_ratio` | bounding rect `w / h` |
| `solidity` | `area / convex_hull_area` |
| `equivalent_diameter` | `√(4A/π)` |
| `mean_intensity` | mean gray value under the contour mask |
| `centroid_x/y` | image moments (fallback: bounding-rect center) |

### 2.5 Serialization + distribution

- `colony_to_feature_dict(colony)` → the 7-feature row consumed by the shape model.
- `colony_measurement_to_json(colony, id, shape)` → rounded JSON row (`advanced` mode).
- `detect_distribution(colonies, image_shape)` → spatial layout via nearest-neighbour
  centroid distances normalized by image diagonal:
  `< 0.08` → `clustered`, `< 0.16` → `mixed`, else `dispersed`; `< 2` colonies → `isolated`.

---

## 3. `training.py` — hierarchical sklearn training pipeline

The 696-line workhorse. Entry point: `train_models()`.

### 3.1 Data loading & label engineering

- `_load_labeled_dataframe(dataset_csv)` — reads the CSV, normalizes organism names,
  keeps only rows whose organism is in `ORGANISM_METADATA`, and **derives**
  `organism_type`, `gram_label`, `shape_label`, `taxonomy_group` from the metadata
  dict. Raises if no supported organisms.
- `_resolve_image_path(workspace_root, image_path)` — tries
  `workspace_root / path`, then rewrites legacy `"Bacteria dataset/..."` → `data/dataset/`,
  then the raw path; raises `FileNotFoundError`.

### 3.2 Feature tables

- `_build_image_feature_table(labeled_df, workspace_root, augment)` — one row per
  image of 616 classical features (+ metadata). When `augment=True`, adds
  `AUGMENT_PER_IMAGE` photometric variants using a deterministic md5-of-path seed and
  tags rows with `variant_id`. **Augmentation never touches the test fold.**
- `_build_image_embedding_table(labeled_df, embedding_model, ...)` — the DL twin:
  one row per image with columns `emb_0..emb_{d-1}`. Called **after** the split so the
  shared holdout stays leakage-free. No photometric augmentation (DL has its own
  transform pipeline).
- `_build_colony_feature_table(labeled_df, workspace_root)` — for every image, detect
  colonies and emit **one row per colony** with the image's `shape_label`
  (**weak label** — every colony in a cocci image is labeled cocci). Images with zero
  detected colonies are skipped.

### 3.3 Splits (leakage discipline, issues #6/#8)

- **Image-level holdout:** `train_test_split(test_size=0.2, random_state=42, stratify=organism)`
  on `labeled_df` in `train_models()`. Both the classical and embedding feature tables
  are built from this same split, so results are comparable and the holdout is shared.
- `_image_level_split(labeled_df, ...)` — the colony-table twin; stratifies on
  `shape_label` (falls back to unstratified when a class has <2 samples).
- `_split_colonies_by_image(...)` — `GroupShuffleSplit` grouped by `image_id` so no
  single plate's colonies leak into both train and test.
- An `assert` verifies train/test image sets are disjoint.

### 3.4 Label cleaning

- `_clean_colony_label_table(colony_table)` — applies `SHAPE_CLEANING_RULES` to drop
  contradictory colony rows; returns removal stats (per-shape counts) for the audit
  trail. Applied to both the full table (for `supported_shape_labels`) and to each
  split's table.

### 3.5 Model selection — `_fit_best_ensemble_model`

The same 80/20 split is reused for every classifier. Three candidates are trained
on the (oversampled) training fold and scored on the holdout; the best wins:

| Candidate | Hyperparameters |
| --------- | --------------- |
| RandomForest | n_estimators=380, class_weight=`balanced_subsample`, n_jobs=-1 |
| ExtraTrees | n_estimators=520, class_weight=`balanced_subsample`, n_jobs=-1 |
| KNN scaled | StandardScaler + KNN(5, weights=`distance`) |

- `_oversample_train(...)` — bootstrap minority classes up to the majority count
  **within the training fold only** (never touches test → no leakage).
- Scoring: `accuracy` (default) or `balanced_accuracy` (used for organism_type).

### 3.6 The classifier stack trained

| Model | Data | Target |
| ----- | ---- | ------ |
| `gram_model` | bacteria rows only | gram_label |
| `organism_type_model` | all rows | organism_type |
| `group_model` | all rows | taxonomy_group |
| `organism_model` | all rows | organism (global species, fallback) |
| `group_species_models[group]` | rows per group | organism (specialist) |
| `shape_model` | colony rows | shape_label |

`_train_group_species_models(...)` trains one specialist per taxonomy group; a group
with a single class gets a `random_forest_single_class` placeholder instead of the
ensemble search.

### 3.7 Metrics + artifacts

- `_classification_summary(...)` → accuracy, full `classification_report`, optional
  `balanced_accuracy` and `per_modality_accuracy`.
- `_per_modality_metrics(...)` → accuracy per `imaging_type` on the holdout
  (**gram-stain vs media-plate** — the known confounder, issue #8).
- Artifact dict saved via `joblib.dump`:

```python
{
  "gram_model", "organism_type_model", "group_model", "organism_model",
  "group_species_models", "shape_model",
  "image_feature_columns", "colony_feature_columns",
  "supported_shape_labels", "supported_organisms", "organism_metadata",
  "feature_version", "training_meta", "model_choices",
  "embedding_model_path": "embedding_model.pt",   # only when --dl
}
```

- Metrics JSON (`*.metrics.json`) written alongside: per-model summaries,
  `per_modality_metrics`, `training_meta`, `model_choices`.

---

## 4. `inference.py` — prediction orchestration + JSON assembly

Entry point: `predict_bacteria_image(image_path, model_path=None, mode="basic")`.

### 4.1 `load_models(model_path)`

`joblib.load` the bundle, then `_attach_embedding_model(artifacts, dir)`: if the
bundle has `embedding_model_path`, resolve it (relative → relative to the joblib
dir), load the `.pt` with `dl/extract.load_embedding_model`, and stash
`_embedding_model`, `_embedding_encoders`, `_embedding_config`, `_embedding_input_size`
into the artifacts dict under private keys.

### 4.2 Image vector construction

- DL present → `extract_embedding(..., input_size=_embedding_input_size)` → 1-row
  DataFrame on `image_feature_columns`; also sets `dl_species_prediction` if the
  species encoder is attached.
- Classical → `extract_image_features(image)` → 616-dim DataFrame on
  `image_feature_columns`.

> Note: `extract_image_features` is still always computed (used by the rejection
> gates for `edge_density` / `laplacian_var`), even on the DL path.

### 4.3 Hierarchical prediction (order matters)

```python
predicted_organism_type = organism_type_model.predict(vec)   # bacteria / fungi
predicted_group         = group_model.predict(vec)           # 4 taxonomy groups
predicted_bacteria_name = <species resolution>               # 10 organisms
gram_label              = gram_model.predict(vec)            # gram_* / non_bacterial_fungi
```

#### Species resolution — two paths

- **DL path** (`dl_species_prediction`): `dl/extract.species_probabilities(..., tta=True)`
  gives softmax probabilities (averaged over original + h-flip + two scale-jitter
  crops), restricted to `ORGANISMS_BY_GROUP[predicted_group]`, argmax wins. This is
  what pushed holdout species accuracy 0.33 → **0.63**.
- **Classical path** `_predict_species_with_group_constraint(...)`: use the group
  specialist model if it exists, else the global `organism_model` restricted to the
  group's organisms (probability re-argmax over candidates).

### 4.4 Metadata consistency override

```python
meta = organism_metadata.get(predicted_bacteria_name, {})
if meta:
    predicted_organism_type = meta["organism_type"]
    gram_label              = meta["gram_label"]
```

The taxonomy table always wins — a classifier can never contradict known biology.

### 4.5 Colony + shape branch (parallel to the hierarchy)

1. `extract_colonies(image)` → per-colony measurements.
2. For each colony: predict shape with the `shape_model`, `_shape_by_heuristic`
   overrides to **`spiral`** when `aspect_ratio ≥ 3.0` and `circularity ≤ 0.35`.
3. `_dominant_shape_and_confidence(preds, probs)` — majority vote; blended confidence
   `0.65·dominance_ratio + 0.35·mean_prob`. If `predicted_organism_type == "fungi"`,
   dominant shape is forced to `"fungal"`.
4. `final_confidence = max(shape_conf, organism_conf)`.

### 4.6 Rejection gate — `_should_reject_image(...)`

Rejects (all label fields → `unknown`, counts still reported) when **any** of:

- `colony_count < MIN_COLONIES_FOR_VALID`
- `organism_type_prob < MIN_ORGANISM_TYPE_PROB`
- both `organism_confidence < MIN_ORGANISM_CONFIDENCE` **and**
  `shape_confidence < MIN_COLONY_CONFIDENCE`
- both `edge_density < MIN_EDGE_DENSITY` **and** `laplacian_var < MIN_LAPLACIAN_VAR`
  (out-of-focus / blank image)

### 4.7 Output contracts

**Basic:**
```json
{ "organism_type", "predicted_bacteria_name", "bacteria_type",
  "total_colonies_detected", "dominant_shape", "confidence" }
```

**Advanced:** adds `total_colonies`, `colonies[]` (per-colony measurements, via
`colony_measurement_to_json`), and `final_morphology` with `distribution` from
`detect_distribution`.

Both have a matching `unknown` variant when rejected.

---

## 5. `dl/` — PyTorch embedding sub-module

The deep-learning redesign (port from `BPA22_clone`). A pretrained CNN backbone +
multi-task heads produces a **learned image embedding** that replaces the 616-dim
hand-crafted vector for the sklearn hierarchy.

### 5.1 `dl/__init__.py` — critical thread pinning

```python
cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

OpenCV and torch each link their own OpenMP thread pool; running both concurrently
caused intermittent memory corruption / segfaults on macOS. Pinning both to
single-threaded CPU is harmless because heavy compute runs on **MPS**. **Do not
remove this.**

Also re-exports `EmbeddingModel`, `build_embedding_model`.

### 5.2 `dl/model.py` — `EmbeddingModel`

```python
class EmbeddingModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True,
                 num_species=10, num_groups=4, num_grams=3, num_types=2)
```

- **Backbone** (`_build_backbone`): strips the classifier head of
  `efficientnet_b0` (1280), `resnet18` (512), `resnet50` (2048), or
  `mobilenet_v3_large` (960); pretrained with `IMAGENET1K_V1` weights.
- **Pool + embedding:** `AdaptiveAvgPool2d(1)` → flatten → the embedding vector.
- **Four linear heads** on the embedding: `species_head`, `group_head`, `gram_head`,
  `type_head` — used **during training** as regularizers and for the species task.
- `forward(x)` → `{"embeddings": emb, "logits": {species, group, gram, type}}`.
- `embedding(x, normalize=False)` — convenience; optional L2-normalization (the
  contrastive loss normalizes internally).
- `freeze_backbone()` / `unfreeze_backbone()` — implement the frozen→fine-tune
  schedule (`requires_grad` toggling).
- `build_embedding_model(pretrained=True, **kwargs)` — factory.

### 5.3 `dl/dataset.py` — `LabelEncoder` + `ImageDataset`

- `LabelEncoder` — deterministic sorted index↔label mapping (`encode`, `decode`,
  `to_dict`, `__len__`). Serialized into the checkpoint so inference can decode.
- `ImageDataset(Dataset)` — wraps a labeled DataFrame; `__getitem__` reads the image
  with `cv2.imread`, applies the transform, and returns `(tensor, targets)` where
  targets is a dict of encoded species/group/gram/type tensors.
- Helpers: `subset(indices)` and `with_transform(transform)` return shallow copies
  (used by the split and by choosing train vs val transforms). `encoders` property
  returns all four encoders.
- `build_image_dataset(table)` — builds all four `LabelEncoder`s from the DataFrame
  and returns the dataset.

### 5.4 `dl/transforms.py` — all-OpenCV transforms

Deliberately **no torchvision tensor ops** (avoids the OpenMP clash).

- `inference_transform(input_size=224)` — resize to `input_size²` (`INTER_AREA`),
  center-crop, BGR→RGB, `×1/255`, ImageNet mean/std normalize. Used for both train
  (augment off) and val, and at inference.
- `train_transform(seed, input_size)` — scale-jitter resize (0.8–1.0), center crop,
  50/50 h-flip + v-flip, ±15° rotation (`warpAffine`), photometric jitter
  (contrast ×0.85–1.15, Gaussian brightness noise σ=0.02). **Off by default** — it
  HURT whole-plate accuracy (0.48 → 0.25), so `TrainingConfig.augment=False`.

### 5.5 `dl/losses.py` — `CombinedLoss`

```python
L = L_species(CE) + aux_weight·(L_group + L_gram + L_type) + contrastive_weight·L_contrastive
```

- `supervised_contrastive_loss(embeddings, labels, temperature=0.1)` — Khosla et al.
  supervised contrastive loss: L2-normalize, cosine similarity matrix / τ, mask self,
  pull together same-species pairs, push apart different-species pairs (log-sum-exp
  denominator). Only computed when `batch_size > 1`.
- `CombinedLoss(aux_weight=0.3, contrastive_weight=0.1, temperature=0.1,
  head_keys=("group","gram","type"))` — returns `(total, parts)` where `parts` holds
  each term for logging. The species head carries the main task; group/gram/type are
  auxiliary regularizers that share the embedding.

### 5.6 `dl/trainer.py` — `fit_embedding_model`

- `TrainingConfig` dataclass: `batch_size=8`, `epochs=30`, `lr=3e-4`,
  `weight_decay=1e-4`, `aux_weight=0.3`, `contrastive_weight=0.1`, `temperature=0.1`,
  `val_fraction=0.2`, `random_state=42`, `frozen_epochs=0`, `use_scheduler=True`,
  `augment=False`, `input_size=224`.
- `get_device()` — `cuda` → `mps` → `cpu` (the M2 trains on MPS).
- `split_dataset(dataset, val_fraction, random_state)` — stratified image-level split
  on **species**, seed 42, **mirroring `training.py`** so DL and classical holdouts
  are comparable. (The 20% holdout in `train_models` is kept separate and untouched.)
- `fit_embedding_model(model, dataset, config, checkpoint_path, device=None)`:
  1. Seed everything (`_set_seed`) including MPS.
  2. Split → assign transforms (augment or not for train, inference for val).
  3. DataLoaders with a custom `_collate` that stacks images + target dicts.
  4. `freeze_backbone()` if `frozen_epochs > 0`; unfreeze when
     `epoch == frozen_epochs`.
  5. AdamW + optional cosine-annealing scheduler (over the fine-tune epochs).
  6. Per epoch: train loop → evaluate species accuracy on val → **save checkpoint on
     improvement** (`val_acc_species >= best_acc`).
  7. Checkpoint content: `model_state_dict`, `label_encoders` (via `to_dict`),
     `config` (backbone, embedding_dim, num_species, epochs, lr, aux/contrastive
     weights, input_size), `val_acc_species`.
- Returns `(history, metadata)` — per-epoch train loss + val species accuracy, device,
  counts, checkpoint path.

### 5.7 `dl/extract.py` — inference-time embedding

- `load_embedding_model(checkpoint_path)` — `torch.load(map_location="cpu",
  weights_only=False)`, rebuild the `EmbeddingModel` from the saved config with
  `pretrained=False`, `load_state_dict`, rebuild `LabelEncoder`s. Returns
  `(model, encoders, config)`.
- `extract_embedding(model, image_path, device="cpu", input_size=224)` — single image
  → 1280-dim tensor (`no_grad`, returns CPU tensor). This is what feeds the sklearn
  hierarchy at inference.
- `species_probabilities(model, image_path, species_encoder, ..., tta=False)` —
  softmax over the species head keyed by species name. With `tta=True`, averages over
  the original, a horizontal flip, and two scale-jitter crops (0.85, 0.7).
- `predict_species(...)` — convenience wrapper returning `(name, confidence)`.

---

## 6. Data flow across modules (training vs inference)

### Training

```
dataset_full.csv
      │  _load_labeled_dataframe (config.ORGANISM_METADATA)
      ▼
 labeled_df ── 80/20 stratified holdout (seed 42)
      │                                  │
      ├─(no --dl)→ _build_image_feature_table → 616-dim rows
      └─(--dl)  → _build_image_embedding_table → emb_0..emb_d-1 rows
                                          │
       (separate) _build_colony_feature_table → per-colony 7-dim rows
                                          │
   _fit_best_ensemble_model (RF/ET/KNN) on each target (training.py)
                                          │
                    joblib.dump artifact bundle (+ .metrics.json)
```

### Inference

```
image ──► load_models (attach .pt if present)
   │
   ├─► extract_image_features (always, for quality gates)
   ├─► (DL?) extract_embedding : extract_image_features  → image_vector
   │
   │  organism_type_model → group_model → species (group-constrained, DL head or
   │  sklearn) → gram_model
   │
   │  extract_colonies → shape_model per colony (+ spiral heuristic)
   │
   ▼
   metadata override → dominant shape + confidence → rejection gate
                                          │
                                   JSON (basic / advanced / unknown)
```

---

## 7. Versioning & backward compatibility

- `FEATURE_VERSION = 3` is embedded in the artifact's `training_meta` and metrics —
  bump it on any feature change so stale artifacts are caught.
- A bundle **without** `embedding_model_path` falls back to the full classical path
  (616-dim features + sklearn species model). Removing the `.pt` degrades accuracy
  but the system still runs.
- The `.pt` filename is stored relative to the joblib dir; inference resolves it
  relative to the artifact location, so the bundle stays portable.

---

## 8. Quick reference — public API

| Module | Symbol | Purpose |
| ------ | ------ | ------- |
| `__init__` | `predict_bacteria_image`, `train_models` | package exports |
| `config` | `ORGANISM_METADATA`, `ORGANISMS_BY_GROUP`, thresholds, `FEATURE_VERSION` | taxonomy + knobs |
| `features` | `extract_image_features`, `extract_colonies`, `detect_distribution`, `augment_image` | classical features + CV |
| `training` | `train_models` | train + save bundle + metrics |
| `inference` | `predict_bacteria_image`, `load_models` | predict + JSON |
| `dl.model` | `EmbeddingModel`, `build_embedding_model` | CNN + multi-task heads |
| `dl.dataset` | `ImageDataset`, `LabelEncoder`, `build_image_dataset` | data loading |
| `dl.transforms` | `inference_transform`, `train_transform` | OpenCV transforms |
| `dl.losses` | `CombinedLoss`, `supervised_contrastive_loss` | training objectives |
| `dl.trainer` | `fit_embedding_model`, `TrainingConfig`, `get_device` | training loop |
| `dl.extract` | `load_embedding_model`, `extract_embedding`, `species_probabilities` | inference-time embedding |
