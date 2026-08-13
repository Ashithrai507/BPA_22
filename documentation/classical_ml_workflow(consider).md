# Classical ML Workflow — Complete End-to-End Walkthrough

This document explains, in full detail, how the classical machine-learning system
in `src/bacteria_assistant/` works: how raw images become numeric features, how the
hierarchical classifiers are trained and invoked, how a single image is classified,
and how the final JSON output is assembled.

Everything below is traced from the actual code:

| Concern        | File                                  |
| -------------- | ------------------------------------- |
| Taxonomy/consts | `src/bacteria_assistant/config.py`    |
| Features       | `src/bacteria_assistant/features.py`  |
| Training       | `src/bacteria_assistant/training.py`  |
| Inference      | `src/bacteria_assistant/inference.py` |
| CLI training   | `train_model.py`                      |
| CLI inference  | `predict_bacteria.py`                 |

---




## 1) The Big Picture

```
 dataset_full.csv  ──►  labeled DataFrame  ──►  image features  ──►  hierarchical models  ──►  artifacts/joblib
                          (629 images)             (616 dims)            (6 model groups)
                                                                                │
 input image ──► global features ──► hierarchy inference ──► colony segmentation ──► colony features ──► shape model
                                                                                │
                                                                                ▼
                                                            aggregation + consistency + confidence
                                                                                │
                                                                                ▼
                                                          JSON output (basic / advanced / unknown)
```

The system is **classical ML** — no deep learning. Images are converted into
hand-engineered numeric vectors, and `scikit-learn` tree/nearest-neighbour models
predict on those vectors. There is no gradient training; the models are fitted once
in `train_models()` and serialized with `joblib`.

---

## 2) Taxonomy and Constants (config.py)

Every supported organism is mapped to its biological labels:

| Organism              | organism_type | gram_label            | shape_label | taxonomy_group          |
| --------------------- | ------------- | --------------------- | ----------- | ----------------------- |
| Staphylococcus aureus | bacteria      | gram_positive         | cocci       | gram_positive_cocci     |
| Streptococcus pyogenes| bacteria      | gram_positive         | cocci       | gram_positive_cocci     |
| Enterococcus faecalis | bacteria      | gram_positive         | cocci       | gram_positive_cocci     |
| Bacillus subtilis     | bacteria      | gram_positive         | bacilli     | gram_positive_bacilli   |
| Clostridium sporogenes| bacteria      | gram_positive         | bacilli     | gram_positive_bacilli   |
| Escherichia coli      | bacteria      | gram_negative         | bacilli     | gram_negative_bacilli   |
| Klebsiella pneumoniae | bacteria      | gram_negative         | bacilli     | gram_negative_bacilli   |
| Pseudomonas aeruginosa| bacteria      | gram_negative         | bacilli     | gram_negative_bacilli   |
| Candida albicans      | fungi         | non_bacterial_fungi   | fungal      | fungi                   |
| Aspergillus niger     | fungi         | non_bacterial_fungi   | fungal      | fungi                   |

Relevant constants:

- `SUPPORTED_SHAPES = ("cocci", "bacilli", "spiral", "fungal")`
- `ORGANISMS_BY_GROUP` — restricts which species are allowed inside each group
  (used during inference to constrain species predictions).
- Rejection thresholds (`MIN_*`) — gate whether the output is a real prediction
  or `"unknown"` (see §9).

---

## 3) Feature Extraction (features.py)

There are **two kinds of features**: image-level (for the species/gram/group models)
and colony-level (for the shape model).

### 3.1 Global image features — `extract_image_features(image)`

Every image is first resized to **256×256** (INTER_AREA), then converted to RGB, HSV
and grayscale. The resulting vector is **616 numbers**:

| Group | Features | Count |
| ----- | -------- | ----- |
| RGB mean + std | `r_mean, r_std, g_mean, g_std, b_mean, b_std` | 6 |
| HSV mean + std | `h_mean, h_std, s_mean, s_std, v_mean, v_std` | 6 |
| Grayscale       | `gray_mean, gray_std`                            | 2 |
| Texture         | `laplacian_var` (sharpness), `edge_density` (fraction of Canny edge pixels) | 2 |
| Color histograms| 8-bin normalized histogram per RGB channel       | 24 |
| Spatial signature| 24×24 downsampled grayscale, flattened, normalized by 255 | 576 |
| **Total**       |                                                    | **616** |

The **spatial signature** (576 values) is the strongest discriminator — it encodes
the coarse layout/texture of the plate. Color statistics capture stain tone; the
histograms and Laplacian/Canny capture texture. These 616 values are the fixed
`image_feature_columns` used to train and run every image-level classifier.

### 3.2 Colony segmentation — `extract_colonies(image)`

1. Convert image to grayscale.
2. Gaussian blur `(5,5)`.
3. Run Otsu threshold **twice** — binary and binary-inverted — because colonies can
   be darker or lighter than the plate background.
4. Morphological open then close with a `3×3` kernel (removes specks, fills holes).
5. `cv2.findContours(RETR_EXTERNAL)`.
6. **Area filtering**: keep only contours with `area >= max(20, 0.00003 × image_area)`
   and `area <= 0.12 × image_area`.
7. Pick the mask (inverted vs plain) that yields the **most valid contours**
   (`_choose_best_mask`); if only one mask produced contours, use it.
8. Sort colonies by area, largest first.

Each accepted contour becomes a `ColonyMeasurement` with **7 geometric features +
centroid**:

```
area, perimeter, circularity = 4πA/P²,
aspect_ratio = bbox_width / bbox_height,
solidity = area / convex_hull_area,
equivalent_diameter = √(4A/π),
mean_intensity (mean gray inside contour),
centroid_x, centroid_y
```

`detect_distribution(colonies, image_shape)` classifies the layout using
nearest-neighbour distances normalized by the image diagonal:

| Mean NN distance / diagonal | Label |
| --------------------------- | ----- |
| < 0.08                      | `clustered` |
| < 0.16                      | `mixed`     |
| otherwise                   | `dispersed` |
| < 2 colonies                | `isolated`  |

---

## 4) Training Pipeline (training.py)

Entry point: `train_models(dataset_csv, workspace_root, model_output_path)`.

### 4.1 Label preparation — `_load_labeled_dataframe`
- Read `dataset_full.csv` (rows: `image_path`, `organism`, `imaging_type`, …).
- Normalize organism names (`"Bacillus subtilis_gram stain"` → `"Bacillus subtilis"`)
  and keep only the 10 known organisms.
- Map each organism to `organism_type`, `gram_label`, `shape_label`, `taxonomy_group`
  using `ORGANISM_METADATA`.

### 4.2 Training tables
- `_build_image_feature_table`: for each of the **629** labeled images, extract the
  616 global features + labels. → `image_table`.
- `_build_colony_feature_table`: for each training image, segment colonies and
  produce one row of 7 colony features per colony, **all labelled with the image's
  `shape_label`** (i.e. colony-level ground truth is inherited from the parent image).
  This yields **45,507 colony rows**.

### 4.3 Model auto-selection — `_fit_best_ensemble_model`
For every classifier task, **three candidates** are trained on a stratified
80/20 train/test split and the one with the best **test accuracy** wins:

1. `RandomForestClassifier(n_estimators=380, class_weight="balanced_subsample")`
2. `ExtraTreesClassifier(n_estimators=520, class_weight="balanced_subsample")`
3. `KNN (n=5, distance)` wrapped in a `StandardScaler` pipeline

The chosen model and its metrics are stored. For groups with a single species
(`y_group.nunique() < 2`) a plain RandomForest is trained with no validation.

### 4.4 The six trained model components

| Artifact key        | Task                                                        | Training rows |
| ------------------- | ----------------------------------------------------------- | ------------- |
| `organism_type_model` | bacteria vs fungi                                          | 629 |
| `group_model`       | taxonomy group (4 classes)                                  | 629 |
| `organism_model`    | global species (10 classes, fallback)                       | 629 |
| `group_species_models` | one specialist species model **per taxonomy group**      | per group |
| `gram_model`        | gram_positive vs gram_negative (bacteria rows only)         | 506 |
| `shape_model`       | cocci / bacilli / fungal (colony features)                  | 45,507 |

### 4.5 Saved artifact bundle
`joblib.dump` writes to `artifacts/bacteria_models.joblib`:
models + `image_feature_columns` + `colony_feature_columns` + supported labels +
`ORGANISM_METADATA` + `training_meta` + `model_choices`.
Metrics are also written to `artifacts/bacteria_models.metrics.json`.

**Measured accuracy of the current artifact:** organism_type 0.82, gram 0.64,
group 0.45, species 0.33, shape 0.68.

---

## 5) Inference — The Classification Flow (inference.py)

Entry point: `predict_bacteria_image(image_path, model_path, mode="basic"|"advanced")`.

The flow below is executed **once per call** in exactly this order.

### Step 1 — Load bundle + read image
- `load_models()` deserializes the joblib artifact.
- `read_image()` reads via `np.fromfile` + `cv2.imdecode` (robust to spaces/unicode).

### Step 2 — Compute the image vector
- `extract_image_features(image)` → 616-dim dict.
- Convert to a one-row DataFrame re-indexed to the saved `image_feature_columns`
  (guarantees column order matches training).

### Step 3 — Organism type (bacteria vs fungi)
- `organism_type_model.predict(image_vector)` → `predicted_organism_type`.
- `predict_proba` max → `organism_type_prob` (used later for rejection).

### Step 4 — Taxonomy group
- `group_model.predict(image_vector)` → one of the 4 groups,
  e.g. `gram_positive_cocci`.

### Step 5 — Species prediction (group-constrained)
`_predict_species_with_group_constraint(artifacts, organism_model, vector, group)`:

1. Look up the **group-specialist model** for the predicted group.
   - If found: predict species with it → return `(species, max proba)`.
2. Otherwise (fallback): use the **global species model**, then pick the species
   **with the highest probability among those in `ORGANISMS_BY_GROUP[group]`**.
   - If the global model lacks `predict_proba`, return its raw prediction with 0.5.

### Step 6 — Gram label
- `gram_model.predict(image_vector)` → `gram_label` (gram_positive/gram_negative).

### Step 7 — Metadata override (consistency)
- Look up `ORGANISM_METADATA[predicted_bacteria_name]`.
- If the species is known, **override** `predicted_organism_type` and `gram_label`
  with the metadata truth. (This is the biological sanity net: a species whose
  group-model label is known forces correct organism/gram output.)
- E.g. fungi species ⇒ `gram_label = "non_bacterial_fungi"`.

### Step 8 — Colony segmentation
- `extract_colonies(image)` → list of `ColonyMeasurement`, sorted by area.

### Step 9 — Per-colony shape prediction
For each colony (id 1..N):
1. Build a one-row DataFrame from `colony_to_feature_dict` on `colony_feature_columns`.
2. `shape_model.predict(row_df)` → `pred` (cocci/bacilli/fungal).
3. If the model supports probabilities, record `max(proba)` as the colony confidence.
4. **Spiral heuristic** — `_shape_by_heuristic`:
   `aspect_ratio >= 3.0 and circularity <= 0.35` ⇒ override shape to `"spiral"`.
   (Spiral has no training data; it is inferred purely from geometry.)
5. Append `(pred, prob)` to running lists and build the advanced-mode JSON row.

### Step 10 — Morphology aggregation — `_dominant_shape_and_confidence`
- `dominant_shape` = most frequent colony shape (`Counter.most_common(1)`).
- `dominance_ratio = dominant_count / total_colonies`.
- `mean_prob` = mean of per-colony probabilities.
- `shape_confidence = 0.65 × dominance_ratio + 0.35 × mean_prob`, clamped to [0,1].
- **Fungi override:** if `predicted_organism_type == "fungi"` ⇒ `dominant_shape = "fungal"`.
- `final_confidence = max(shape_confidence, organism_confidence)`.

### Step 11 — Rejection / uncertainty gate — `_should_reject_image`
The image is rejected (output becomes `"unknown"`) if **any** of:

| Condition | Meaning |
| --------- | ------- |
| `colony_count < 1` | no colonies segmented |
| `organism_type_prob < 0.60` | organism-type classifier is unsure |
| `organism_confidence < 0.55` **and** `shape_confidence < 0.45` | both models disagree strongly |
| `edge_density < 0.015` **and** `laplacian_var < 25.0` | image too flat/out-of-focus |

### Step 12 — Output assembly (§6 below)

---

## 6) Output Contracts

### 6.1 Basic mode
```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies_detected": 42,
  "dominant_shape": "bacilli",
  "confidence": 0.81
}
```

### 6.2 Advanced mode
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
`final_morphology.distribution` comes from `detect_distribution` (§3.2).

### 6.3 Rejected / unknown output
When §Step 11 fires, every label becomes `"unknown"` (colony counts are still
reported):
```json
{
  "organism_type": "unknown",
  "predicted_bacteria_name": "unknown",
  "bacteria_type": "unknown",
  "total_colonies_detected": 0,
  "dominant_shape": "unknown",
  "confidence": 0.72
}
```
(advanced mode mirrors this with `total_colonies`, `colonies`, and
`final_morphology` where `distribution` is also `"unknown"`).

---

## 7) Worked Example (single image end-to-end)

Input: `"Bacteria dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png"`

1. Read → BGR ndarray.
2. Resize 256×256 → 616 global features.
3. `organism_type_model` → `bacteria` (proba 0.91) ✅ above 0.60 threshold.
4. `group_model` → `gram_positive_bacilli`.
5. Group specialist for `gram_positive_bacilli` predicts `Bacillus subtilis`
   (proba 0.84) → `organism_confidence = 0.84`.
6. `gram_model` → `gram_positive`; metadata confirms → `bacteria` /
   `gram_positive`.
7. Segmentation finds 42 colonies.
8. Shape model labels colonies mostly `bacilli` (high aspect ratio, low
   circularity); one elongated contour trips the spiral heuristic → `spiral`.
9. `dominant_shape = bacilli`, `dominance_ratio = 40/42 = 0.95`,
   `mean_prob = 0.80` → `shape_confidence = 0.65×0.95 + 0.35×0.80 = 0.90`.
10. `final_confidence = max(0.90, 0.84) = 0.90`.
11. Rejection gate passes (0 colonies check, proba 0.91, confidences high, image sharp).
12. Output assembled — basic or advanced depending on `mode`.

---

## 8) Hierarchical Decision Tree

```
image_vector (616-dim)
│
├─ organism_type_model ──► bacteria │ fungi
│                             │
├─ group_model ──────────────► gram_positive_cocci │ gram_positive_bacilli │ gram_negative_bacilli │ fungi
│                             │
├─ group_species_models[group] ─► species (group-constrained)
│      └─ (fallback) organism_model restricted to ORGANISMS_BY_GROUP[group]
│
├─ gram_model (bacteria only) ──► gram_positive │ gram_negative
│      └─ overridden by ORGANISM_METADATA[specles] for consistency
│
└─ shape_model per colony ──► cocci │ bacilli │ fungal  (+ spiral heuristic)
       └─ aggregate ──► dominant_shape + confidence + distribution
```

Why hierarchy? Each classifier sees a smaller decision space, which reduces
biologically invalid outputs (e.g. a gram-negative species inside a gram-positive
group) and isolates errors.

---

## 9) Key Implementation Notes

- **Species fallback**: the global `organism_model` is only used when no group
  specialist exists; even then, candidates are restricted to the predicted group's
  species.
- **Consistency over model output**: metadata lookups (`ORGANISM_METADATA`) always
  win over raw model predictions for organism/gram when the species is known.
- **Spiral is synthetic**: produced only by the geometry heuristic, never by the
  shape model (no spiral samples exist in the dataset).
- **Fungi dominance**: any fungi image is forced to `dominant_shape = "fungal"`.
- **Confidence blend**: `final_confidence` is the *maximum* of shape and species
  confidence, while per-colony shape confidence is a *weighted blend* of class
  dominance and mean model probability.
- **UI runs inference twice** (`basic` + `advanced`) per Analyze click
  (`bacteria_ui.py` `_predict`), so a single analysis performs Steps 1–12 twice.
- The trained artifact in `artifacts/` was produced in a different workspace
  (`BPA22_clone`); retrain locally with `python train_model.py` to refresh paths
  and metrics.

---

## 10) Running It

```bash
# Train
python train_model.py

# Predict (basic or advanced)
python predict_bacteria.py --image "path/to/image.png" --mode basic
python predict_bacteria.py --image "path/to/image.png" --mode advanced

# GUI
python bacteria_ui.py

# Tests (output contract)
python -m pytest -q
```
