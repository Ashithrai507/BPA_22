# Image → Output: Complete Processing Pipeline

This document describes exactly what happens, step by step, when a microscope image
is fed into the system: the inputs, every system/component that runs in between, and
the final output produced.

## 1. Big picture

```
                     ┌───────────────────────────────────────────────────────---───┐
                     │                      ENTRY POINTS                           │
                     |CLI: predict_bacteria.py --image <img> --mode basic|advanced │
                     |GUI: bacteria_ui.py  (upload image + click "Predict")        │
                     └───────────────────────────┬──────────────────────────----───┘
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │          STEP A  —  LOAD & PREPARE ARTIFACTS   m         │
                    │  load_models(bacteria_models.joblib)                     │
                    │    • joblib bundle: sklearn hierarchy + metadata         │
                    │    • DL embedding model (embedding_model.pt) if present  │
                    └───────────────────────────┬──────────────────────────────┘
                                                ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │        STEP B  —  READ & DECODE THE INPUT IMAGE          │
                    │  read_image(): np.fromfile + cv2.imdecode (BGR ndarray)  │
                    └───────────────────────────┬──────────────────────────────┘
                                                ▼
              ┌──────────────────────────────┬──────────────────────────────┐
              ▼                              ▼                              ▼
  ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
  │  STEP C1 — IMAGE-LEVEL   │  │  STEP C2 — COLONY-LEVEL  │  │  STEP C3 — QUALITY INFO  │
  │  (DL or classical)       │  │  (classical CV, kept)    │  │  (edge/laplacian, kept)  │
  │  embedding / feature vec │  │  Otsu + contours →       │  │  for future rejection    │
  │  → 1280-dim or 616-dim   │  │  per-colony measurements │  │  gates                   │
  └────────────┬─────────────┘  └────────────┬─────────────┘  └──────────────────────────┘
               ▼                             ▼
  ┌──────────────────────────┐  ┌──────────────────────────┐
  │  STEP C1a — HIERARCHY    │  │  STEP C2a — SHAPE MODEL  │
  │  organism_type → group → │  │  per-colony 7 features   │
  │  species (group-          │  │  → cocci/bacilli/spiral  │
  │  constrained) + gram      │  │  + heuristic override    │
  └────────────┬─────────────┘  └────────────┬─────────────┘
               ▼                             ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │        STEP D  —  ASSEMBLY & POST-PROCESSING             │
                    │  metadata consistency overrides                          │
                    │  dominant shape vote + confidence blend                  │
                    │  distribution detection (clustered/mixed/dispersed)      │
                    └───────────────────────────┬──────────────────────────────┘
                                                 ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │             STEP E  —  JSON OUTPUT CONTRACT              │
                    │  mode=basic    → flat summary object                     │
                    │  mode=advanced → summary + per-colony list + morphology  │
                    └──────────────────────────────────────────────────────────┘
```

---

## 2. Entry points (how an image gets in)

| Entry point | Command / action | Mode |
| ----------- | ---------------- | ---- |
| CLI | `python predict_bacteria.py --image <path> --mode basic` | basic |
| CLI | `python predict_bacteria.py --image <path> --mode advanced` | advanced (default) |
| GUI | `python bacteria_ui.py` → File upload → Predict | toggleable details |

Both call the exact same function: `predict_bacteria_image(image_path, model_path, mode)`
in `src/bacteria_assistant/inference.py`.

---

## 3. Step A — Load the model artifact

`load_models()` loads `artifacts/bacteria_models.joblib`:

- The **sklearn hierarchy** (trained on DL embeddings):
  - `organism_type_model` — bacteria vs fungi
  - `group_model` — 4 taxonomy groups (gram_positive_cocci, gram_positive_bacilli,
    gram_negative_bacilli, fungi)
  - `organism_model` — global species classifier (10 organisms)
  - `group_species_models` — specialist species model per taxonomy group
  - `gram_model` — gram_positive / gram_negative / non_bacterial_fungi
  - `shape_model` — per-colony shape (cocci / bacilli / spiral / fungal)
- `image_feature_columns` — the column layout of the feature vector
  (`emb_0`…`emb_1279` for DL artifacts, 616 classical columns otherwise)
- `organism_metadata` — the fixed biological taxonomy table (species → type/gram/shape/group)
- **Optional DL model**: if the artifact points to `embedding_model.pt`, the
  EfficientNet-B0 embedding model is loaded too. Old artifacts without it work via
  graceful fallback to classical features.

> Models used: scikit-learn ensembles (RandomForest / ExtraTrees / KNN+StandardScaler,
> best selected per task at training time) + PyTorch EfficientNet-B0 for the embedding.

---

## 4. Step B — Read the input image

- `read_image()` reads the raw bytes with `numpy.fromfile` (handles spaces/unicode
  paths) and decodes with `cv2.imdecode` into a **BGR ndarray**.
- Supported formats: PNG, JPG, BMP, TIFF, WEBP.

From here the pipeline forks into two parallel branches: image-level analysis and
colony-level analysis.

---

## 5. Step C1 — Image-level analysis

### 5.1 Feature vector: DL embedding (primary path)

When the artifact bundles `embedding_model.pt`:

1. Image is resized/cropped to the trained `input_size` (320×320 after tuning).
2. `extract_embedding()` forwards the image through EfficientNet-B0 → global average
   pool → **1280-dim L2-normalized embedding vector**.
3. The vector is wrapped in a 1-row DataFrame using `image_feature_columns`.

*Fallback (classical artifacts):* `extract_image_features()` computes a **616-dim**
hand-engineered vector instead — color stats (RGB/HSV/gray), Laplacian variance,
Canny edge density, color histograms, and a 24×24 gray spatial signature.

### 5.2 Hierarchical classification on that vector

1. **organism_type** — `organism_type_model.predict()` → `bacteria` | `fungi`
2. **taxonomy group** — `group_model.predict()` → one of the 4 groups
3. **species** — two possible paths:
   - **DL path (primary):** softmax over the DL species head with **test-time
     augmentation** (original + horizontal flip + scale-jitter crops, averaged),
     then restricted to organisms allowed inside the predicted group
     (`ORGANISMS_BY_GROUP`), then argmax → species name + confidence.
   - **Classical path:** group-specialist sklearn model when one exists for the
     predicted group; otherwise the global organism model restricted to the
     predicted group's organisms.
4. **gram label** — `gram_model.predict()` → gram_positive / gram_negative /
   non_bacterial_fungi

> This "predict group first, then species only within that group" constraint is what
> prevents biologically impossible cross-group species predictions and lifts species
> accuracy from 0.33 (baseline) to 0.63 (DL).

---

## 6. Step C2 — Colony-level analysis (classical CV, unchanged)

Runs in parallel with the image-level branch:

1. Convert to grayscale.
2. **Otsu thresholding** (inverted + plain masks, both tried) with morphological
   open/close.
3. `cv2.findContours` → external contours, filtered by size (min/max area relative
   to image).
4. The mask yielding the richer, more realistic contour set is chosen.
5. Each contour → `ColonyMeasurement` (sorted largest first):
   - `area`, `perimeter`, `circularity`, `aspect_ratio`, `solidity`,
     `equivalent_diameter`, `mean_intensity`, `centroid_x/y`

### 6.1 Per-colony shape

- 7 geometry features → `shape_model.predict()` → cocci / bacilli / spiral / fungal.
- **Heuristic override:** contours with `aspect_ratio ≥ 3.0` and `circularity ≤ 0.35`
  are forced to `spiral` (spiral has no training labels; produced by morphology
  heuristics per README).

---

## 7. Step D — Assembly & post-processing

1. **Metadata consistency override** — `organism_metadata[species]` overrides
   `organism_type` and `gram_label`, so the biological taxonomy is never contradicted
   by a classifier.
2. **Dominant shape** — majority vote across all colony predictions.
3. **Confidence blend** — `0.65 × dominance_ratio + 0.35 × mean_shape_prob`, and the
   final confidence is `max(shape_confidence, species_confidence)`.
4. **Distribution** (advanced only) — mean nearest-neighbour distance between colony
   centroids normalized by image diagonal → `clustered` (<0.08) / `mixed` (<0.16) /
   `dispersed`, or `isolated` when <2 colonies.
5. Fungi images force `dominant_shape = "fungal"`.

---

## 8. Step E — JSON output contracts

### Basic mode

```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies_detected": 42,
  "dominant_shape": "cocci",
  "confidence": 0.81
}
```

### Advanced mode

```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies": 42,
  "colonies": [
    {
      "id": 1,
      "area": 109.5,
      "perimeter": 45.69,
      "circularity": 0.65,
      "aspect_ratio": 1.02,
      "solidity": 0.89,
      "equivalent_diameter": 11.8,
      "mean_intensity": 132.4,
      "predicted_shape": "cocci"
    }
  ],
  "final_morphology": {
    "dominant_shape": "cocci",
    "distribution": "clustered",
    "confidence": 0.87
  }
}
```

---

## 9. Which system/component produces what

| Output field | Produced by |
| ------------ | ----------- |
| `organism_type` | `organism_type_model` on the embedding vector, overridden by `organism_metadata` |
| `predicted_bacteria_name` | DL species head + TTA, group-constrained (or sklearn group-specialist) |
| `bacteria_type` | `gram_model` on the embedding vector, overridden by `organism_metadata` |
| `total_colonies_detected` | Classical Otsu + contours (`extract_colonies`) |
| `colonies[].*geometry` | `_contour_to_measurement` (OpenCV contour math) |
| `colonies[].predicted_shape` | `shape_model` on 7 geometry features + spiral heuristic |
| `final_morphology.dominant_shape` | Majority vote of colony shapes |
| `final_morphology.distribution` | `detect_distribution` (nearest-neighbour centroid analysis) |
| `confidence` | Blend of shape dominance ratio + shape/species probabilities |

---

## 10. Key source files

| File | Role |
| ---- | ---- |
| `predict_bacteria.py` | CLI entry point |
| `bacteria_ui.py` | PyQt5 GUI entry point |
| `src/bacteria_assistant/inference.py` | Orchestration: `predict_bacteria_image` |
| `src/bacteria_assistant/features.py` | `read_image`, 616-dim classical features, colony detection/measurement |
| `src/bacteria_assistant/dl/extract.py` | DL embedding extraction + TTA species probabilities |
| `src/bacteria_assistant/config.py` | Taxonomy metadata, groups, model path |
| `src/bacteria_assistant/training.py` | Training of the sklearn hierarchy + artifact bundle |
| `artifacts/bacteria_models.joblib` | Trained models bundle |
| `artifacts/embedding_model.pt` | Trained EfficientNet-B0 embedding model |
