# BPA-22 — Concept Q&A

> This file captures two clarification conversations about the `bacteria_assistant/`
> pipeline, with the full answers as given. It records *why* the architecture works
> the way it does, in Q&A form. Read it alongside
> [`bacteria_assistant_modules.md`](bacteria_assistant_modules.md) and
> [`developer.md`](developer.md).

---

## Q1 — "A pretrained CNN backbone + multi-task heads produces a learned image embedding that replaces the 616-dim hand-crafted vector for the sklearn hierarchy" — what does this actually mean?

This sentence packs a lot of meaning. Unpacking it piece by piece:

### 1. The two "worlds" of features

A sklearn classifier (RandomForest, etc.) can't look at pixels directly — it needs
**one row of numbers per image**. The question is *where those numbers come from*.

**Classical (hand-crafted):** a human writes formulas. `extract_image_features` in
`features.py:66` computes 616 numbers by hand: color means/std, histograms, a 24×24
spatial grid, edge density, Hu moments, etc. The strengths/weaknesses:

- The features encode what a human *guessed* matters (color, texture, layout).
- Cheap, fast, no GPU, fully transparent.
- They miss patterns a human didn't think of, which is why species accuracy topped
  out at ~0.33.

**Deep-learning (learned):** instead of hand-writing the formulas, we *let a neural
network figure out which numbers matter*, then read out those numbers as the
"embedding."

### 2. What a CNN backbone does

A CNN processes an image through stacked convolutional layers that learn a
**hierarchy of patterns**:

```
pixels → edges → textures/shapes → object parts → biological features
layer 1      layer 2       layer 3          layer 4
```

In `dl/model.py:75-84`:

```python
features = self.pool(self.backbone(x))   # (B, C, 1, 1) — pooled feature maps
emb = torch.flatten(features, 1)         # (B, 1280)    — the embedding
```

The `AdaptiveAvgPool2d(1)` collapses the final feature maps into a single
**1280-number summary** of "what this network sees in the image." Those 1280 numbers
*are* the embedding. Each dimension is some learned, usually-unnameable visual
property — but critically, the training process shapes them so that they're *useful
for the prediction tasks*.

### 3. Why "pretrained" matters

`build_embedding_model` uses `IMAGENET1K_V1` weights (`dl/model.py:52`). The backbone
starts with weights already trained on ImageNet (millions of photos of everyday
objects). It doesn't start from zero — it already knows how to detect edges, textures,
and shapes. The bacterial fine-tuning (frozen → fine-tune schedule in
`dl/trainer.py:135-137`) adapts those general visual skills to bacterial colonies.
This is **transfer learning** — you couldn't train 5M parameters from scratch on only
~629 images.

### 4. Why "multi-task heads"

Four linear layers sit on top of the embedding (`dl/model.py:45-48`):

```python
self.species_head = nn.Linear(self.embedding_dim, num_species)  # 10 species
self.group_head   = nn.Linear(self.embedding_dim, num_groups)   # 4 taxonomy groups
self.gram_head    = nn.Linear(self.embedding_dim, num_grams)    # 3 gram labels
self.type_head    = nn.Linear(self.embedding_dim, num_types)    # bacteria/fungi
```

During training, `CombinedLoss` (`dl/losses.py`) optimizes all four heads plus a
contrastive term. Why does this help the *embedding*? Because a good embedding must
simultaneously be useful for **species**, **group**, **gram**, and **type**
prediction. It's like a student who must pass four exams — the knowledge they retain
is richer and more general than if they studied for only one. The embedding becomes a
shared representation where same-species images are close together (pulled by the
contrastive loss) and different ones are far apart.

> Crucially, at inference only the **embedding** is consumed downstream; the heads
> (except the species head for species prediction) were just training scaffolding
> that made the embedding good.

### 5. "Replaces the 616-dim vector for the sklearn hierarchy"

This is the key architectural insight: **the sklearn stack is unchanged** — only its
input changes.

- **Before (classical):** `train_models` builds a table where each image's row is 616
  hand-crafted numbers (`_build_image_feature_table`, `training.py:75`). The sklearn
  classifiers train on those columns.
- **After (DL):** each image's row is the 1280 numbers read out of the CNN
  (`_build_image_embedding_table`, `training.py:113`, columns named `emb_0..emb_1279`).
  The *exact same* `_fit_best_ensemble_model` trains the *exact same* hierarchy
  (gram → organism_type → group → species → shape) on those columns.

```python
# inference.py:149-154 — the vector is built from the CNN instead of hand-crafted features
emb = extract_embedding(embedding_model, Path(image_path), input_size=input_size)
emb_cols = artifacts["image_feature_columns"]
image_vector = pd.DataFrame([dict(zip(emb_cols, emb.tolist(), strict=False))])[emb_cols]
```

So the sklearn models don't know or care whether a number came from a hand-written
formula or a neural net — they just see "1280 real-valued features per image." That's
the "replacement."

### 6. Why keep the sklearn hierarchy at all?

Because the CNN is a **smarter feature extractor, not a full system**. The hierarchy
on top provides what raw CNN output lacks:

1. **Taxonomy structure** — group-constrained species prediction, metadata overrides
   (`inference.py:218-221`) make biologically impossible outputs impossible.
2. **Interpretability & contracts** — stable JSON output, confidence gating.
3. **Small-data friendliness** — the 10-way species head alone plateaued at ~0.48;
   constraining it to the predicted group pushed it to 0.63.

### 7. The measurable payoff

From `developer.md` (fixed holdout, seed 42):

| Modality | Classical (616-dim) | DL (1280-dim embedding) |
| -------- | ------------------- | ----------------------- |
| species | 0.33 | **0.63** |
| group | 0.45 | **0.91** |
| organism_type | 0.82 | **0.99** |
| gram | 0.64 | **0.92** |

Same classifiers, same split, same task — the only change is the numbers fed in.
That's the entire thesis of the DL redesign.

---

## Q2 — Do classical and deep learning work simultaneously, or does only one work at a time?

Short answer: **not both at the same time — it's an either/or switch** for the image
features, but the overall system is *always* a hybrid of both.

### 1. The image-feature path: one OR the other

This is decided **per artifact bundle** (per `.joblib` you load), not per image. The
switch is `embedding_model_path` inside the bundle:

```python
# inference.py:147-159
embedding_model = artifacts.get("_embedding_model")     # None if no .pt attached
if embedding_model is not None:
    image_vector = extract_embedding(...)   # DL: 1280-dim CNN embedding
    dl_species_prediction = True
else:
    image_vector = pd.DataFrame([image_features])  # classical: 616-dim
    dl_species_prediction = False
```

- **DL bundle** (`bacteria_models.joblib`, trained with `--dl`): every image's row is
  the 1280-dim CNN embedding → DL species head.
- **Classical bundle** (`bacteria_models_classical.joblib`, trained without `--dl`):
  every image's row is the 616-dim hand-crafted vector → sklearn species model.

It's a pure if/else. One image goes through exactly one of these paths. If you load a
DL bundle, the classical feature extractor is *not* used for classification.

### 2. But the system is still "hybrid" — different roles, always both

"Classical vs DL" only describes the *image-feature extractor*. In every prediction,
both kinds of ML are used — just for different jobs:

| Component | What runs | ML type | Always? |
| --------- | --------- | ------- | ------- |
| Image → feature vector | `extract_embedding` (CNN) | **DL** | only on DL bundle |
| Image → feature vector | `extract_image_features` | classical | only on classical bundle |
| organism_type / group / gram | sklearn hierarchy | classical | **always** |
| species | DL head (group-constrained) | **DL** | only on DL bundle |
| species | sklearn group-specialist | classical | only on classical bundle |
| colony detection + measurements | OpenCV Otsu/contours | classical CV | **always** |
| colony shape | sklearn `shape_model` | classical | **always** |
| quality gates (edge/laplacian) | `extract_image_features` | classical | **always even on DL path** |

Two interesting details:

- **On the DL path, `extract_image_features` still runs** — but only to feed the
  rejection gates (`edge_density`, `laplacian_var`), not the classifiers
  (`inference.py:253-260`).
- **Colony/shape never involves DL at all** — that branch is 100% classical in both
  bundles.

### 3. The one-sentence takeaway

> At any instant, the sklearn hierarchy consumes **either** the 1280-dim CNN
> embedding **or** the 616-dim classical vector — never both mixed — but every
> prediction simultaneously uses classical CV (colonies), classical sklearn
> (gram/group/type/shape), and *optionally* the CNN (features + species).

So the original sentence — *"a learned embedding replaces the 616-dim hand-crafted
vector"* — means: for the DL artifact, the vector is swapped out entirely. The
replacement is total, not additive.

---

## Q3 — Do we use decision trees? How do we classify bacteria type and shape when there are multiple bacteria?

### 1. Yes, we use decision trees — as ensembles

The three candidate classifiers in `_fit_best_ensemble_model` (`training.py:356-383`):

| Candidate | What it is |
| --------- | ---------- |
| **RandomForest** (n_estimators=380) | an *ensemble* of decision trees |
| **ExtraTrees** (n_estimators=520) | an *ensemble* of randomized decision trees |
| **KNN scaled** | not a tree — nearest neighbours on standardized features |

Every model (gram, type, group, species, shape) is auto-picked from these 3 by
holdout accuracy — so the final artifact is usually a RandomForest or ExtraTrees,
i.e. **hundreds of decision trees voting together**. A single tree overfits and is
weak; hundreds averaged together are strong and stable. So: yes, decision trees (as
random forests / extra-trees) are the workhorse, with KNN as a candidate.

### 2. How multi-class works when there are 10 bacteria

Two important points:

**a) Decision trees handle multi-class natively.** You don't need "one model per
bacteria." A single decision tree splits on features (e.g. `if emb_412 > 0.7 → group
A`), and at each leaf it can output **any of the classes**. The Gini/entropy split
criterion directly measures multi-class impurity. A RandomForest then trains hundreds
of trees and the class with the most votes wins. So 10 classes is just a label set of
size 10 — no magic needed.

**b) But we don't use one flat 10-class model — we use a hierarchy.** Multiple
classifiers, each with its own label set:

| Model | Classes | Trained on | Predicts |
| ----- | ------- | ---------- | -------- |
| `organism_type_model` | 2 (bacteria, fungi) | all images | organism type |
| `group_model` | 4 taxonomy groups | all images | group |
| `organism_model` | **10** (global species, fallback) | all images | species |
| `group_species_models[group]` | 3 / 2 / 3 / 2 per group | that group's images only | species (specialist) |
| `gram_model` | 3 (gram_positive, gram_negative, non_bacterial_fungi) | **bacteria rows only** | gram label |
| `shape_model` | cocci / bacilli / fungal | per-colony rows | shape |

So the classification for **bacteria type (gram)** and **species** flows like this:

```
organism_type_model  → bacteria or fungi?
   │
group_model          → gram_positive_cocci / gram_positive_bacilli /
   │                    gram_negative_bacilli / fungi
   │
species: restrict candidates to ORGANISMS_BY_GROUP[predicted_group]
   │      then run the group-specialist model (or DL species head)
   │
gram_model (bacteria rows only) → gram_positive or gram_negative
   │
metadata override: ORGANISM_METADATA[bacteria] fixes type + gram from the
   │                biology table (inference.py:218-221)
```

**Example for "Staphylococcus aureus":**

```
image → type_model: "bacteria"
      → group_model: "gram_positive_cocci"
      → species: only candidates = {Staph aureus, Strep pyogenes, Enterococcus faecalis}
                 specialist model picks "Staphylococcus aureus"
      → gram_model: "gram_positive"
      → override: metadata confirms gram_positive + cocci
```

**Shape** is a separate, parallel branch (`inference.py:231-246`): each detected
colony becomes 7 numbers (area, perimeter, circularity, aspect_ratio, solidity,
diameter, intensity) → `shape_model` predicts cocci/bacilli/fungal per colony →
majority vote = dominant shape → `spiral` if geometry says so (`aspect_ratio ≥ 3.0`,
`circularity ≤ 0.35`).

### 3. The key insight

The hierarchy shrinks the decision space at each step. Instead of one model facing 10
similar-looking bacteria, each specialist faces only 2–3, and the group constraint
makes biologically impossible outputs (coccus predicted as gram-negative bacillus)
impossible. That's why species accuracy jumped 0.33 → 0.63.

---

## Q4 — Basic doubt: what happens when a user uploads an image? (walkthrough in depth)

The full flow, traced through the actual code (`scripts/bacteria_ui.py` +
`src/bacteria_assistant/inference.py`).

### Part A — Uploading the image (Open Image...)

When the user clicks **Open Image...** (`_choose_image`, `bacteria_ui.py:580`):

1. **File dialog** opens filtered to `png/jpg/jpeg/bmp/tif/tiff/webp`.
2. The chosen path is stored in `self.image_path`.
3. **Backend validation** (`bacteria_ui.py:595`): `read_image()` (`features.py:57`)
   reads the file via `np.fromfile` + `cv2.imdecode` — the same reader used later for
   prediction. If it fails → a warning dialog and the Analyze button stays disabled.
   This guarantees "if you can select it, the backend can read it."
4. **Preview** (`bacteria_ui.py:602-613`): `QPixmap` loads the file; if that fails
   (some files only open via the cv2 backend), it falls back to converting the numpy
   BGR array → RGB → `QImage` → `QPixmap` (`_preview_pixmap_from_array`). The image is
   scaled to fit the viewer with `KeepAspectRatio`.
5. UI state flips to "loaded": Analyze button enabled, result fields reset to "—",
   status light → orange "Image loaded".

> Nothing has been predicted yet at this point — just file read + preview.

### Part B — Clicking Analyze (`_predict`, `bacteria_ui.py:636`)

1. **Model check:** verifies `artifacts/bacteria_models.joblib` exists; if missing →
   "Please run train_model.py first."
2. Status → "Analyzing...", `QApplication.processEvents()` to repaint.
3. **Calls `predict_bacteria_image` TWICE** (`bacteria_ui.py:654-655`) — once with
   `mode="basic"` and once with `mode="advanced"`. Each call is a full independent
   pipeline run (both load the ~650 MB joblib bundle and the 16 MB `.pt` from disk).
4. On exception → critical error dialog.
5. Fills the 5 result rows: NAME, TYPE, SHAPE, COLONIES, CONFIDENCE. Confidence is
   color-coded (`bacteria_ui.py:669-676`): **green** > 0.75, **orange** > 0.5,
   **red** otherwise.
6. `_detailed_payload` = `{input_image, basic_output, advanced_output}` is
   pretty-printed into the hidden QTextEdit; **Show Details** button enables.

### Part C — What `predict_bacteria_image` actually does (for EACH of the two calls)

The full backend pipeline (`inference.py:135-309`):

1. **Load models** — `load_models` (`inference.py:33-39`): `joblib.load` the bundle
   (gram/type/group/species/shape models + column lists + metadata).
   `_attach_embedding_model` (`inference.py:42-59`): if the bundle has
   `embedding_model_path`, loads the `.pt` checkpoint (16 MB) and its label encoders,
   stored under private keys. This decides **which world we're in**: DL or classical.

2. **Read the image** — `read_image` → numpy BGR array.

3. **Build the image vector** (`inference.py:147-159`):
   - **DL bundle:** `extract_embedding` → 1280-dim CNN embedding → 1-row DataFrame on
     `image_feature_columns`.
   - **Classical bundle:** `extract_image_features` → 616-dim vector → same DataFrame.
   - Note: `extract_image_features` is *still* run either way — its
     `edge_density`/`laplacian_var` feed the rejection gate later.

4. **The hierarchy** (one row in, labels out) (`inference.py:170-216`):

   ```
   organism_type_model.predict(vec)  → bacteria / fungi     (line 170)
   group_model.predict(vec)          → 4 taxonomy groups    (line 182)
   species resolution (line 191-213):
      DL: species_probabilities(tta=True) restricted to
          ORGANISMS_BY_GROUP[predicted_group] → argmax
      classical: group-specialist model (or global model restricted to group)
   gram_model.predict(vec)           → gram_positive/negative (line 216)
   ```

5. **Metadata consistency override** (`inference.py:218-221`):
   `ORGANISM_METADATA[species]` overwrites `organism_type` and `gram_label` — the
   biology table always wins over the classifiers.

6. **Colony + shape branch** (parallel) (`inference.py:223-251`):
   - `extract_colonies` → Otsu thresholding → contours → 7 measurements per colony.
   - Per colony: `shape_model.predict` → spiral heuristic (`aspect_ratio ≥ 3.0`,
     `circularity ≤ 0.35`).
   - `_dominant_shape_and_confidence`: majority vote + blended confidence; forced to
     `fungal` if organism type is fungi.

7. **Rejection gate** (`inference.py:253-260`): rejects → all labels `unknown` if too
   few colonies, low type probability, low confidence, or blurry image (edge/Laplacian
   gates).

8. **JSON assembly**:
   - `basic`: `{organism_type, predicted_bacteria_name, bacteria_type,
     total_colonies_detected, dominant_shape, confidence}`
   - `advanced`: same + `total_colonies`, `colonies[]` (per-colony measurements),
     `final_morphology{dominant_shape, distribution, confidence}`

Both results go back to the UI, which renders the summary rows and stores the JSON for
"Show Details."

### Notable behavior

The GUI runs inference **twice** (basic + advanced), so each Analyze click re-loads
~666 MB of models and runs the full CV+ML pipeline two times. That's why Analyze takes
a few seconds.

---

## Q5 — The full pipeline from the beginning: data collection → preprocessing → feature extraction → training → inference

A complete end-to-end discussion, grounded in the actual data (measured from
`data/dataset/dataset_full.csv` and the real artifacts).

### 1. Data collection — what was gathered

**Raw inputs:** 629 microscope images of bacterial/fungal cultures, all
**640×640×3 BGR uint8** (confirmed by reading real files).

**The taxonomy being predicted** (from `config.py`): 10 organisms, each with a fixed
biology — organism type (bacteria/fungi), gram stain, shape, and taxonomy group.

**Class balance (measured from `dataset_full.csv`):**

| Organism | Images |
| -------- | ------ |
| Staphylococcus aureus | 91 |
| Streptococcus pyogenes | 72 |
| Aspergillus niger | 64 |
| Escherichia coli | 61 |
| Pseudomonas aeruginosa | 60 |
| Candida albicans | 59 |
| Bacillus subtilis | 57 |
| Klebsiella pneumoniae | 57 |
| Clostridium sporogenes | 55 |
| Enterococcus faecalis | 53 |

Imbalanced (91 → 53) — which is *why* the training code oversamples and the DL model
was trained from only ~629 images.

**The confounder built into the data:** every organism exists in **two imaging
modalities** — 325 gram-stain + 304 media-plate images. These look very different
(lighting/background, not biology). This drives several design decisions later (CLAHE,
per-modality metrics, the modality-confounder tests).

**Messy folder names** (from `data/dataset/`): `Staph aureus`, `Pseudomonas auergosa`,
`Aspergillosis niger` — typos/abbreviations in *folder* names while the `organism`
column uses canonical names. `normalize_organism_name` + `_load_labeled_dataframe`
handle this.

### 2. Data preprocessing — normalization + label engineering

**The CSV → labeled DataFrame** (`_load_labeled_dataframe`, `training.py:55`):

1. Read `dataset_full.csv` (one row per image).
2. `normalize_organism_name` collapses underscores/whitespace.
3. Keep only rows whose organism matches `ORGANISM_METADATA` keys (drops unknowns).
4. **Derive all 4 label columns** from the biology table — never stored in the CSV:
   - `organism_type` (bacteria/fungi)
   - `gram_label` (gram_positive/gram_negative/non_bacterial_fungi)
   - `shape_label` (cocci/bacilli/fungal)
   - `taxonomy_group` (4 groups)

**Path resolution** (`_resolve_image_path`, `training.py:36`): CSVs reference
`"Bacteria dataset/..."` but images live under `data/dataset/` — a relocation
fallback handles the mismatch.

### 3. Feature extraction — two worlds

**3a. Classical 616-dim** (`extract_image_features`, `features.py:66`):
the image is resized to 256×256, then:

| Group | Count | What |
| ----- | ----- | ---- |
| RGB stats | 6 | r/g/b mean + std |
| HSV stats | 6 | h/s/v mean + std |
| gray stats | 2 | gray_mean/std — **gray is CLAHE-normalized** |
| texture | 2 | Laplacian variance (focus), Canny edge density |
| RGB histograms | 24 | 8-bin per-channel |
| **spatial signature** | 576 | 24×24 grayscale grid → plate layout |
| colony Hu moments | 22 | whole-mask + per-contour means/stds |
| gradient texture | 3 | Sobel magnitude mean/std, angle dispersion |

**Preprocessing inside:** CLAHE (`clipLimit=2.0`, 8×8 tiles) applied *before*
spatial/edge/Laplacian features — this is the anti-confounder step (gram-stain vs
media-plate lighting).

**3b. Colony features 7-dim** (`extract_colonies`, `features.py:235`): per image,
CLAHE gray → GaussianBlur(5×5) → Otsu threshold (both inverted & plain) → morph
open/close → contours → **pick the richer mask** → per contour: area, perimeter,
circularity (4πA/P²), aspect_ratio, solidity, equivalent_diameter, mean_intensity +
centroid.

**3c. DL 1280-dim embedding** (`dl/extract.py`): image → resize to 320² → center crop
→ BGR→RGB → /255 → ImageNet mean/std normalize → EfficientNet-B0 → adaptive avg pool →
1280-dim vector. The pretrained backbone + multi-task heads (species/group/gram/type)
+ contrastive loss train the embedding so it's useful for all tasks.

### 4. Training splits (leakage discipline)

- Shared **80/20 stratified holdout on `organism`**, `random_state=42`
  (`training.py:480`) — the SAME split for classical and DL feature tables.
- Colony rows: `_image_level_split` stratifies on `shape_label`;
  `_split_colonies_by_image` uses `GroupShuffleSplit` grouped by `image_id` so no
  plate leaks (issue #6).
- **Oversampling** minority classes within the training fold only
  (`_oversample_train`).
- Colony label cleaning via `SHAPE_CLEANING_RULES` (issue #10) — drops contradictory
  geometry rows.

### 5. Training the models

`_fit_best_ensemble_model` tries 3 candidates per task (RandomForest 380, ExtraTrees
520, KNN-scaled) → best by holdout accuracy. The actual DL artifact chose (from the
real `bacteria_models.metrics.json`):

| Model | Winner | Holdout accuracy |
| ----- | ------ | ---------------- |
| organism_type | KNN scaled | **0.992** |
| group | ExtraTrees | **0.905** |
| gram | KNN scaled | **0.922** |
| organism (global species) | KNN scaled | 0.484 |
| group species specialists | RandomForest | (varies) |
| shape | RandomForest | **0.675** |

The 0.484 flat species score is exactly why the group-constrained DL species head
(0.63) replaced it at inference.

### 6. Inference (what happens at predict time)

Covered in depth in Q4: load bundle (+ optional `.pt`) → image vector (DL or
classical) → hierarchy type→group→species→gram → metadata override → colony/shape
branch → rejection gate → JSON.

### 7. The pipeline in one picture

```
collection (629 × 640×640 BGR images, 10 organisms, 2 modalities)
   │
preprocessing: normalize names → derive labels from ORGANISM_METADATA
   │
feature extraction: 616-dim classical  OR  1280-dim DL embedding
   │                    + 7-dim per-colony features
   │
splits (seed 42): 80/20 image-level holdout, image-grouped colonies, oversample train
   │
training: per-task best of {RandomForest, ExtraTrees, KNN} → artifact bundle
   │
inference: hierarchy → overrides → gates → JSON (basic/advanced/unknown)
```

---

## Quick reference — files cited

| File | Symbols cited |
| ---- | ------------- |
| `scripts/bacteria_ui.py` | `_choose_image` (580), `_preview_pixmap_from_array` (572), `_predict` (636), `_toggle_details` (628) |
| `src/bacteria_assistant/features.py` | `read_image` (57), `extract_image_features` (66), `extract_colonies` (235) |
| `src/bacteria_assistant/training.py` | `_load_labeled_dataframe` (55), `_resolve_image_path` (36), `_build_image_feature_table` (75), `_build_image_embedding_table` (113), `_fit_best_ensemble_model` (356-383), `_oversample_train` (276), `_image_level_split` (177), `_split_colonies_by_image` (234) |
| `src/bacteria_assistant/inference.py` | `load_models` (33), `_attach_embedding_model` (42), `predict_bacteria_image` (135), vector switch (147-159), metadata override (218-221), colony branch (223-251), rejection gates (253-260) |
| `src/bacteria_assistant/dl/model.py` | `EmbeddingModel` heads (45-48), backbone weights (52), forward/pool (75-84) |
| `src/bacteria_assistant/dl/losses.py` | `CombinedLoss` |
| `src/bacteria_assistant/dl/trainer.py` | frozen→fine-tune schedule (135-137) |
