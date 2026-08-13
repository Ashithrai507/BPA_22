# Developer Guide — Building the Bacteria Prediction System from Scratch

> This is the full, end-to-end developer walkthrough: if you were handed this project
> with nothing but a folder of microscope images, these are the exact steps you take,
> the exact code that runs at each stage, and the exact decisions you make — from raw
> data to a working prediction system with a UI.

---

## 0. The 30-second mental model

You are building a **3-tier classifier system**:

```
        raw microscope images (629)                 ← DATA
        ─────────────────────────────────────────────────
        tier 0: DL embedding (EfficientNet-B0, PyTorch)
                → 1280-dim vector per image            ← FEATURES
        ─────────────────────────────────────────────────
        tier 1: sklearn hierarchy on the vector
                organism_type → group → species + gram  ← MODEL
        tier 2: classical OpenCV colony detection
                → per-colony geometry → shape model     ← CV
        ─────────────────────────────────────────────────
        JSON output (basic / advanced) + PyQt5 UI       ← OUTPUT
```

The biological taxonomy (species → type / gram / shape / group) is **fixed** in
`config.py` and drives both label engineering and inference overrides.

---

## 1. Phase A — Data Collection

### 1.1 What data you need

For each organism you want to predict, collect **whole-plate or gram-stain microscopy
images**. The reference dataset uses **10 organisms**:

| Organism | Type | Gram | Shape | Taxonomy group |
| -------- | ---- | ---- | ----- | -------------- |
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

> **Why these 10?** They cover every combination of gram stain and morphology (cocci,
> bacilli, fungi) with two modalities each. This is your **taxonomy contract** — it
> lives in `src/bacteria_assistant/config.py` (`ORGANISM_METADATA`, `ORGANISMS_BY_GROUP`).

### 1.2 Two imaging modalities (important)

Every organism has images from two modalities:

| Modality | Purpose |
| -------- | ------- |
| **gram stain** | what Gram staining looks like (325 images) |
| **media plate** | what the culture plate looks like (304 images) |

This modality split is a **known confounder** (images differ by lighting/background,
not just biology). Keep it tracked — it is used for per-modality accuracy analysis.

### 1.3 Folder conventions

Use one folder per organism + modality, with a **consistent naming scheme** so the
filenames are self-describing:

```
Bacteria dataset/
├── dataset_full.csv        ← master metadata (one row per image)
├── dataset_labels.csv      ← per-folder class index + counts
├── dataset_summary.csv     ← same as labels, alternative view
├── Bacillus subtilis_gram stain/
│   ├── Bacillus subtilis_gram stain_1.png
│   ├── Bacillus subtilis_gram stain_2.png
│   └── ...
├── Bacillus subtilis_media plate/
│   └── ...
└── ... (20 folders = 10 organisms × 2 modalities)
```

### 1.4 The three CSV files

These CSVs are the **single source of truth** the training code reads:

**`dataset_full.csv`** — one row per image, columns:

| Column | Example | Role |
| ------ | ------- | ---- |
| `image_path` | `Bacteria dataset/.../stain_12.png` | relative path used by all loaders |
| `folder_name` | `Aspergillus niger_gram stain` | the source folder |
| `class_index` | `0` | numeric class (0–19 = organism×modality) |
| `organism` | `Aspergillus niger` | **the label** (must match `ORGANISM_METADATA` keys) |
| `imaging_type` | `gram stain` / `media plate` | modality |
| `standardized_class_name` | `Aspergillus niger_gram stain` | normalized folder name |
| `filename` | `..._12.png` | basename |

**`dataset_labels.csv`** — per folder: `folder_name`, `class_index`, `organism`,
`imaging_type`, `standardized_class_name`, `image_count`.

**`dataset_summary.csv`** — same information; used for quick class-balance checks.

> **Data hygiene rule:** paths must remain relative to the workspace root. The loader
> does `workspace_root / image_path` and only falls back to the raw path if that
> doesn't exist (`_resolve_image_path` in `training.py`).

---

## 2. Phase B — Environment Setup

### 2.1 Python version

The project targets **Python 3.14** (the venv at `.venv/`). PyTorch 2.13.0 /
torchvision 0.28.0 have cp314 macOS arm64 wheels.

### 2.2 Dependencies

`requirements.txt` pins:

| Package | Purpose |
| ------- | ------- |
| numpy 2.2.x | array math |
| pandas 2.2.x | metadata / feature tables |
| opencv-python-headless 4.11 | image I/O, CV, transforms |
| scikit-learn 1.6.1 | classical classifiers + split |
| joblib 1.4.2 | model artifact serialization |
| PyQt5 5.15.11 | desktop GUI |
| pytest 8.3.5 | test runner |
| torch 2.13.0 | DL backbone (embedding) |
| torchvision 0.28.0 | pretrained backbones |

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 2.3 Project layout to create

```
BPA22_clone/
├── Bacteria dataset/            # raw images + CSVs
├── test_data/                   # held-out images for manual testing
├── artifacts/                   # trained models (gitignored *.joblib, .pt tracked)
├── docs/                        # architecture + step log
├── documentation/               # presentation + this guide
├── src/bacteria_assistant/
│   ├── __init__.py
│   ├── config.py                # taxonomy, groups, model path
│   ├── features.py              # read_image, classical features, colony CV
│   ├── training.py              # classical + hierarchical training
│   ├── inference.py             # predict_bacteria_image (the orchestrator)
│   └── dl/
│       ├── model.py             # EmbeddingModel (backbone + 4 heads)
│       ├── dataset.py           # ImageDataset + LabelEncoder
│       ├── transforms.py        # all-OpenCV resize/crop/normalize
│       ├── losses.py            # CombinedLoss (CE + contrastive)
│       ├── trainer.py           # fit_embedding_model (MPS/CPU)
│       ├── extract.py           # embed any image at inference time
│       └── train_dl.py          # CLI (also at repo root)
├── predict_bacteria.py          # CLI entry point
├── train_model.py               # classical + hierarchical training CLI
├── train_dl.py                  # DL embedding training CLI
├── bacteria_ui.py               # PyQt5 GUI
├── tests/
├── requirements.txt
└── pyproject.toml               # pytest pythonpath=src, ruff config
```

### 2.4 pyproject.toml

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff.lint.per-file-ignores]
"bacteria_ui.py" = ["BLE001"]
```

> `pythonpath = ["src"]` is critical — tests and scripts import `bacteria_assistant.*`.

---

## 3. Phase C — Data Cleaning

The cleaning step is **partly manual (inspect the folders) and partly programmatic**
(the loaders normalize aggressively). What a developer must do:

### 3.1 Manual checks

1. **Every image must decode.** Corrupt/truncated files must be removed or replaced.
   `read_image()` uses `np.fromfile` + `cv2.imdecode` precisely because normal
   `cv2.imread` breaks on spaces/unicode paths.
2. **Organism names consistent.** Folders like `Staph aureus`, `Pseudomonas auergosa`,
   `Aspergillosis niger` are the *folder* names; the `organism` column must use the
   canonical names. `normalize_organism_name()` collapses underscores/spacing but the
   **case and wording must match `ORGANISM_METADATA` keys**.
3. **Both modalities present** for every organism (class balance matters).
4. **No duplicates / no garbage files** (`.DS_Store`, icons) inside image folders.

### 3.2 Programmatic normalization (`_load_labeled_dataframe`)

This is what the code does automatically — you must guarantee it survives:

```python
df.loc[:, "organism"] = df["organism"].map(normalize_organism_name)   # trim/underscores→spaces
df.loc[:, "imaging_type"] = df["imaging_type"].astype(str).str.strip().str.lower()
labeled_df = df[df["organism"].isin(ORGANISM_METADATA.keys())].copy()  # drop unknown organisms
```

Then the label columns are **derived, not stored**: for each organism the code looks
up `ORGANISM_METADATA` and adds `organism_type`, `gram_label`, `shape_label`,
`taxonomy_group`. This guarantees label consistency — the taxonomy lives in exactly
one place.

> **Check:** if `labeled_df.empty`, the dataset has zero supported organisms — you've
> mislabeled your CSVs.

### 3.3 Image-level sanity

Before training, spot-check feature extraction runs on a handful of images:

```python
from bacteria_assistant.features import read_image, extract_image_features
img = read_image("Bacteria dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png")
f = extract_image_features(img)
assert len(f) == 616          # classical feature count
```

---

## 4. Phase D — Label Engineering (the taxonomy)

**This is the single most important design decision in the project.**

Instead of asking one model to learn everything, define a **strict biological
taxonomy** and derive all labels from it. In `config.py`:

```python
ORGANISM_METADATA = {
    "Staphylococcus aureus": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "cocci",
        "taxonomy_group": "gram_positive_cocci",
    },
    # ... 10 organisms ...
}

ORGANISMS_BY_GROUP = {
    "gram_positive_cocci": ["Staphylococcus aureus", "Streptococcus pyogenes", "Enterococcus faecalis"],
    "gram_positive_bacilli": ["Bacillus subtilis", "Clostridium sporogenes"],
    "gram_negative_bacilli": ["Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa"],
    "fungi": ["Candida albicans", "Aspergillus niger"],
}
```

### 4.1 Why 4 label targets?

| Target | Classes | Used for |
| ------ | ------- | -------- |
| `organism_type` | bacteria, fungi | top-level sanity |
| `gram_label` | gram_positive, gram_negative, non_bacterial_fungi | required output field |
| `shape_label` | cocci, bacilli, fungal (spiral is heuristic) | colony shape prediction + dominant shape |
| `taxonomy_group` | 4 groups | **group-constrained species prediction** |

### 4.2 The group constraint (why it beats a flat classifier)

A flat 10-way species classifier made biologically impossible errors (e.g. a
gram-positive coccus predicted as a gram-negative bacillus). The fix: predict the
group first, then **restrict species candidates to that group's organisms**. This is
done in `inference.py` (`ORGANISMS_BY_GROUP.get(predicted_group, [])`).

> Result: species holdout accuracy 0.33 → 0.63. Never skip this.

---

## 5. Phase E — Feature Extraction (two worlds)

### 5.1 Classical features (baseline / fallback) — `features.py`

`extract_image_features(image)` builds a **616-dim** vector:

| Group | Count | What |
| ----- | ----- | ---- |
| RGB stats | 6 | r/g/b mean + std |
| HSV stats | 6 | h/s/v mean + std |
| gray stats | 2 | gray_mean, gray_std |
| texture | 2 | `laplacian_var`, `edge_density` |
| RGB histograms | 24 | 8-bin hist per channel |
| **gray spatial signature** | 576 | 24×24 downsampled grayscale grid |

> The 24×24 spatial signature (576 dims) was the single biggest classical
> accuracy boost — it encodes gross plate layout.

### 5.2 DL embeddings (primary) — `dl/extract.py`

An **EfficientNet-B0** (ImageNet-pretrained) backbone + global average pool produces
a **1280-dim embedding**:

```python
features = self.pool(self.backbone(x))     # (B, C, 1, 1)
emb = torch.flatten(features, 1)           # (B, 1280)
```

Pipeline: `read_image` → `inference_transform` (resize to `input_size`, center crop,
BGR→RGB, /255, ImageNet mean/std normalize) → forward → embedding.

At training time the model has 4 heads (species/group/gram/type) on top of the
embedding, but **at inference only the embedding + species head are used**; the
downstream sklearn hierarchy consumes the embedding as if it were the 616-dim vector.

### 5.3 Colony features (per-contour, classical) — `features.py`

Each detected contour becomes 7 numbers (`colony_to_feature_dict`):

`area`, `perimeter`, `circularity`, `aspect_ratio`, `solidity`,
`equivalent_diameter`, `mean_intensity`.

---

## 6. Phase F — DL Embedding Training (tier 0)

### 6.1 Command

```bash
.venv/bin/python train_dl.py \
  --dataset-csv "Bacteria dataset/dataset_full.csv" \
  --backbone efficientnet_b0 \
  --epochs 36 --frozen-epochs 12 \
  --input-size 320 --batch-size 8 \
  --output artifacts/embedding_model.pt
```

### 6.2 What happens inside (`fit_embedding_model`)

1. **Load + label-encode** every image path via `build_image_dataset` (species /
   group / gram / type encoders are derived from `ORGANISM_METADATA`).
2. **Split:** 80/20 stratified on **species**, `random_state=42` — *the same split
   used by the classical training so results are comparable*. The 80% is further
   split into train/val internally (also 80/20, seed 42).
3. **Transform:** `inference_transform` for both train and val by default.
   Augmentation (`train_transform`: scale-jitter crop, flips, ±15° rotation,
   photometric jitter) exists but **HURTS on whole-plate images** — measured
   val species 0.48 → 0.25, so it's off by default (`augment=False`).
4. **Frozen→fine-tune schedule:** first `frozen_epochs` only train the heads
   (`freeze_backbone()`), then `unfreeze_backbone()` and train everything.
5. **Loss** (`CombinedLoss`):
   ```
   L = L_species(CE) + 0.3·(L_group + L_gram + L_type) + 0.1·L_contrastive
   ```
   `L_contrastive` is the supervised contrastive loss (Khosla et al.) on the
   embedding — same-species images are pulled together, others pushed apart.
6. **Optimizer:** AdamW (lr 3e-4, weight_decay 1e-4). **Scheduler:** cosine
   annealing over the fine-tune epochs.
7. **Checkpoint:** after every epoch, if val species accuracy improves, save
   `model_state_dict`, `label_encoders`, `config` (backbone, embedding_dim,
   input_size, epochs, lr, weights) to `artifacts/embedding_model.pt`.

### 6.3 Known hyperparameter findings (from the tuning log)

| Lever | Finding |
| ----- | ------- |
| Augmentation | Hurts whole-plate images; off by default |
| Resolution | 224 → 320 helped (0.48 → 0.54); 384 worse |
| Frozen/fine-tune | 12 frozen + 24 fine-tune sweet spot |
| Aux/contrastive weights | 0.3/0.1 default is fine; 0.1/0.05 no gain |
| TTA at inference | small gain (~+0.01) |

### 6.4 Device

`get_device()`: cuda → mps → cpu. On the M2 (16GB) the project trains on **MPS**.
Batch size must stay modest (8 used).

> **Python 3.14 / OpenCV+torch segfault fix:** both libs link their own OpenMP
> thread pools. `dl/__init__.py` pins `cv2.setNumThreads(0)` and
> `torch.set_num_threads(1)`. Don't remove this.

---

## 7. Phase G — Classical Hierarchical Training (tier 1)

### 7.1 Command

```bash
# classical baseline:
.venv/bin/python train_model.py

# DL embeddings:
.venv/bin/python train_model.py --dl artifacts/embedding_model.pt
```

### 7.2 Inside `train_models()`

1. `_load_labeled_dataframe` → cleaned, label-enriched DataFrame.
2. **Build the image table** (one row per image):
   - `--dl` given → `_build_image_embedding_table`: extract 1280-dim embedding per
     image, columns `emb_0..emb_1279`.
   - no `--dl` → `_build_image_feature_table`: 616 classical features per image.
3. **Model selection helper** `_fit_best_ensemble_model` tries three candidates on
   the **same 80/20 split (seed 42, stratified on the target label)** and picks the
   best by holdout accuracy:

   | Candidate | Hyperparams |
   | --------- | ----------- |
   | RandomForest | n_estimators=380, balanced_subsample, n_jobs=-1 |
   | ExtraTrees | n_estimators=520, balanced_subsample, n_jobs=-1 |
   | KNN scaled | StandardScaler + KNN(5, distance weights) |

4. **Train 5+ models:**

   | Model | Data | Target |
   | ----- | ---- | ------ |
   | `gram_model` | bacteria rows only | gram_label |
   | `organism_type_model` | all | organism_type |
   | `group_model` | all | taxonomy_group |
   | `organism_model` | all | organism (global species) |
   | `group_species_models[group]` | rows per group | organism (within-group specialist) |
   | `shape_model` | colony table | shape_label |

   - Group specialists with a single class get a `random_forest_single_class` model.
   - Non-stratifiable groups fall back to non-stratified splits.
5. **Colony table** (`_build_colony_feature_table`): for every image, detect colonies
   and emit one row per colony with the image's `shape_label` (a.k.a. **weak label** —
   every colony in a cocci image is labeled cocci). Skips images with zero colonies.
6. **Serialization:** `joblib.dump` a dict containing every model + metadata:

   ```python
   artifacts = {
       "gram_model": ..., "organism_type_model": ..., "group_model": ...,
       "organism_model": ..., "group_species_models": {...}, "shape_model": ...,
       "image_feature_columns": ["emb_0", ...], "colony_feature_columns": [...],
       "supported_shape_labels": [...], "supported_organisms": [...],
       "organism_metadata": ORGANISM_METADATA,
       "training_meta": {...}, "model_choices": {...},
       "embedding_model_path": "embedding_model.pt",   # only when --dl
   }
   ```
7. Metrics JSON written alongside: `bacteria_models.metrics.json`.

### 7.3 Expected results

| Modality | Classical | DL | Target |
| -------- | --------- | -- | ------ |
| species | 0.33 | **0.63** | ≥0.55 |
| group | 0.45 | **0.91** | ≥0.60 |
| organism_type | 0.82 | **0.99** | ≥0.85 |
| gram | 0.64 | **0.92** | ≥0.72 |
| shape | 0.68 | 0.68 | unchanged |

---

## 8. Phase H — Inference (`predict_bacteria_image`)

This is the heart of the system (`src/bacteria_assistant/inference.py`). Given an
image path + mode, it produces the JSON contract. Sequence:

### 8.1 Load artifacts
`load_models()` → `joblib.load` bundle, then `_attach_embedding_model()`: if
`embedding_model_path` exists, load the `.pt`, its encoders and config
(`_embedding_model`, `_embedding_encoders`, `_embedding_config`,
`_embedding_input_size`).

### 8.2 Read image
`read_image()` → BGR ndarray.

### 8.3 Image vector
- DL present → `extract_embedding(...)` → 1280-dim → 1-row DataFrame on
  `image_feature_columns`.
- else → `extract_image_features` → 616-dim → same DataFrame.

### 8.4 Hierarchical predictions (order matters)

```python
predicted_organism_type = organism_type_model.predict(vec)   # bacteria/fungi
predicted_group         = group_model.predict(vec)           # 4 groups
predicted_bacteria_name = <species, group-constrained>       # 10 organisms
gram_label              = gram_model.predict(vec)            # gram_*/non_bacterial_fungi
```

Species resolution has two paths:
- **DL path:** `species_probabilities(..., tta=True)` — softmax averaged over the
  original image + horizontal flip + two scale-jitter crops (0.85, 0.7); then restrict
  to `ORGANISMS_BY_GROUP[predicted_group]`; argmax.
- **Classical path:** group-specialist model if it exists for the group; otherwise the
  global organism model restricted to the group's organisms.

### 8.5 Metadata consistency override
```python
meta = ORGANISM_METADATA[predicted_bacteria_name]
if meta:
    predicted_organism_type = meta["organism_type"]
    gram_label              = meta["gram_label"]
```
The biology table always wins — classifiers can't contradict the taxonomy.

### 8.6 Colony detection + shape (parallel branch)
`extract_colonies(image)` (see Phase E / §9) → for each colony predict shape, apply
the **spiral heuristic** (`aspect_ratio ≥ 3.0 and circularity ≤ 0.35` → spiral),
collect predictions.

### 8.7 Assembly
- `dominant_shape` = majority vote of colony shapes; **fungi → "fungal"**.
- `confidence` = max of shape confidence and species confidence.
- `distribution` (advanced only) = nearest-neighbour centroid analysis →
  isolated / clustered / mixed / dispersed.

### 8.8 Output contracts (see `documentation/image_processing_pipeline.md`)

- `mode="basic"` → summary object.
- `mode="advanced"` → summary + `colonies[]` + `final_morphology`.

---

## 9. Phase I — Colony Detection (classical CV, tier 2)

`extract_colonies()` in `features.py`:

1. grayscale (`cv2.cvtColor`).
2. GaussianBlur (5×5).
3. **Otsu threshold**, both inverted and plain — producing two candidate masks; each
   gets morphological open + close (3×3 kernel).
4. `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` on both masks.
5. Size filter: `min_area = max(20, image_area×0.00003)`,
   `max_area = image_area×0.12`.
6. **Pick the mask with the richer contour set** (`_choose_best_mask`).
7. Per contour, compute the 7 measurements + centroid (`_contour_to_measurement`),
   sort by area descending.

> The whole system depends on this heuristic segmenter. If images change style
> (new agar, new stain), re-validate colony counts before trusting shape/distribution
> outputs.

---

## 10. Phase J — GUI + CLI (output layer)

### CLI (`predict_bacteria.py`)
```bash
.venv/bin/python predict_bacteria.py \
  --image "test_data/Bacillus subtilis_gram stain_3.png" \
  --mode basic
# --model to point at a different artifact, --mode advanced for full JSON
```

### GUI (`bacteria_ui.py`)
- Retro Win95/98 look (palette in-file).
- Upload → validates with backend `read_image()` → predict → summary view +
  toggleable full JSON details.
- Supported formats: PNG/JPG/BMP/TIFF/WEBP.

```bash
.venv/bin/python bacteria_ui.py
```

---

## 11. Phase K — Testing & Verification (the discipline)

### 11.1 Test layout
| Test file | Covers |
| --------- | ------ |
| `test_output_contract.py` | JSON basic/advanced/unknown schemas |
| `test_embedding_model.py` | backbone/heads/embedding dim |
| `test_dl_dataset.py` | encoders + dataset indexing |
| `test_dl_losses.py` | contrastive + combined loss |
| `test_dl_training.py` | split + tiny smoke train + checkpoint |
| `test_dl_extract.py` | embedding extraction + species probs |
| `test_dl_train_sklearn.py` | embedding table + sklearn retrain path |
| `test_dl_inference.py` | artifact attach, classical fallback, end-to-end |
| `test_modality_metrics.py` | per-modality accuracy reporting |

```bash
.venv/bin/python -m pytest -q      # 24 tests, all green
.venv/bin/python -m ruff check .   # lint clean
```

### 11.2 Manual verification (always do before calling it done)
1. Run the CLI on `test_data/` images in **both modes** — check the JSON is
   schema-valid and biologically plausible.
2. Open the GUI, upload, predict, inspect summary + details.
3. Cross-check species against the group constraint (a bacillus never outputs cocci
   organisms, etc.).

### 11.3 Reporting metrics honestly
The DL figures are measured on the **fixed holdout** (the 20% stratified split,
seed 42). Per-modality accuracy (gram-stain vs media-plate) is the known confounder —
track it separately, never tune on the holdout.

---

## 12. Phase L — Deployment / Artifacts

| Artifact | Contents | Produced by |
| -------- | -------- | ----------- |
| `artifacts/embedding_model.pt` | state_dict + encoders + config (16 MB) | `train_dl.py` |
| `artifacts/bacteria_models.joblib` | sklearn hierarchy bundle (~650 MB) | `train_model.py --dl` |
| `artifacts/bacteria_models.metrics.json` | holdout metrics | `train_model.py` |
| `artifacts/bacteria_models_classical.joblib` | classical-only bundle (baseline) | `train_model.py` (no --dl) |

> `artifacts/*.joblib` is gitignored (huge); `.pt` and metrics are tracked.

**Backward compatibility:** if a joblib bundle has no `embedding_model_path`,
inference silently falls back to the 616-dim classical path. Old artifacts keep
working.

---

## 13. The exact command sequence (cheat sheet)

```bash
# 0. install
.venv/bin/python -m pip install -r requirements.txt

# 1. (optional) verify data loads
.venv/bin/python -c "from bacteria_assistant.training import _load_labeled_dataframe as l; print(len(l('Bacteria dataset/dataset_full.csv')))"

# 2. train DL embedding
.venv/bin/python train_dl.py --input-size 320 --epochs 36 --frozen-epochs 12 --batch-size 8

# 3. retrain sklearn hierarchy on the embeddings
.venv/bin/python train_model.py --dl artifacts/embedding_model.pt

# 4. (optional) classical baseline for comparison
.venv/bin/python train_model.py --output-model artifacts/bacteria_models_classical.joblib

# 5. predict
.venv/bin/python predict_bacteria.py --image "test_data/Bacillus subtilis_gram stain_3.png" --mode basic
.venv/bin/python predict_bacteria.py --image "test_data/Bacillus subtilis_gram stain_5.png" --mode advanced

# 6. GUI
.venv/bin/python bacteria_ui.py

# 7. verify
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

---

## 14. Common failure modes & fixes

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `Image path does not exist` | CSV path not relative to workspace | fix `image_path` in CSV or pass correct `--workspace-root` |
| `No rows found ... for supported organisms` | `organism` column names don't match `ORGANISM_METADATA` | normalize names / fix CSV |
| Prediction error "missing model file" | never trained | run `train_model.py` (step 2–3) |
| Segfault during DL training | OpenMP thread clash (cv2+torch) | ensure `dl/__init__.py` pins threads to 1 |
| Species accuracy stuck ~0.45 | flat classifier / no group constraint | enforce `ORGANISMS_BY_GROUP` restriction at inference |
| Augmentation drops accuracy | whole-plate center-crop destroys signal | keep `augment=False` |
| Colonies wildly wrong on new images | threshold assumptions changed | re-validate Otsu/contour params in `features.py` |

---

## 15. Are we still using the traditional ML? (short answer: yes — it's a hybrid)

**Deep learning did NOT replace the traditional ML.** The system is a hybrid where
DL took over exactly two roles and everything else is still the classical
sklearn + OpenCV pipeline.

### 15.1 What deep learning replaced

1. **Feature extraction.** Instead of the 616-dim hand-crafted vector
   (`extract_image_features`), the system feeds the sklearn hierarchy a **1280-dim
   learned embedding** from EfficientNet-B0. This is the biggest architectural change.
2. **The species classifier.** The sklearn `organism_model` plateaued at ~0.46 holdout
   accuracy. The **DL species head** (with TTA + group constraint) now predicts the
   species, reaching 0.63.

### 15.2 What is still traditional ML / classical CV

| Output field | Model used at inference | Traditional ML? |
| ------------ | ----------------------- | --------------- |
| `predicted_bacteria_name` | DL species head + TTA, group-constrained (`inference.py:163-182`) | No — DL replaced sklearn here |
| `organism_type` | sklearn `organism_type_model` | Yes (RF/ET/KNN) |
| taxonomy `group` | sklearn `group_model` | Yes |
| `bacteria_type` (gram) | sklearn `gram_model` | Yes |
| colony `predicted_shape` / dominant shape | sklearn `shape_model` on 7 geometry features | Yes |
| Colony detection / measurements | OpenCV Otsu + contours | Classical CV (no ML) |
| The *input* to those sklearn models | 1280-dim DL embedding (was 616-dim features) | DL replaced the features |

### 15.3 The training picture

- `train_dl.py` → trains **only** the embedding model (`embedding_model.pt`).
- `train_model.py --dl artifacts/embedding_model.pt` → **retrains the exact same
  sklearn hierarchy** (gram / organism_type / group / organism / group-specialists /
  shape), but on the 1280-dim embeddings instead of the 616-dim features.
- `train_model.py` (no `--dl`) → the **fully classical baseline** (616-dim features +
  sklearn species model) is still produced for comparison and as a fallback.

### 15.4 Graceful degradation

Inference checks for `embedding_model_path` in the joblib bundle:

- present → DL embedding + DL species head path,
- absent → full classical path (616-dim features + sklearn `organism_model`).

So an artifact trained the "old way" keeps working unchanged — the traditional ML
path is preserved in its entirety, just no longer the primary one for species.

> **Takeaway:** DL is a smarter feature extractor + a better species head on top of a
> traditional ML system. If you remove the `.pt`, the system still runs — it just
> falls back to the weaker classical pipeline (species 0.33 vs 0.63).
