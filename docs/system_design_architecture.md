# BPA-22 Deep Learning Redesign — System Design & Architecture

> Working directory: `BPA22_clone`
> Status: DESIGN DRAFT (pre-implementation)
> Last updated: 2026-08-10

---

## 1. Vision & Objectives

**Vision:** Replace the hand-engineered image-feature pipeline with a **deep learning
feature extractor** so the system achieves materially better species/group/gram accuracy,
while preserving every part of the existing system that already works (UI, JSON output
contracts, colony detection, confidence gating, hierarchical classification).

**Success criteria (measurable):**

| Metric | Baseline (classical, holdout) | Target (DL) |
| ------ | ----------------------------- | ----------- |
| Species accuracy | ~0.33 | ≥ 0.55 |
| Group accuracy | ~0.45 | ≥ 0.60 |
| Organism type (bacteria/fungi) | ~0.82 | ≥ 0.85 |
| Gram accuracy | ~0.64 | ≥ 0.72 |
| Colony shape accuracy | ~0.68 | unchanged (classical kept) |

---

## 2. Decided Scope (from brainstorming)

The following decisions were made with the user:

1. **DL covers image-level feature extraction only.** The CNN learns a discriminative
   embedding from raw pixels, replacing the 616-dim hand-engineered vector.
2. **Colony detection stays classical OpenCV** (Otsu + contours). No new annotations
   needed for colonies.
3. **Downstream system stays identical:** hierarchical sklearn classifiers
   (`organism_type` / `group` / `group-specialist species` / `gram`), confidence
   gating, metadata consistency overrides, JSON basic/advanced contracts, PyQt5 UI,
   output-contract tests.
4. **Training runs locally on Apple M2 (16 GB)** using PyTorch MPS/CPU.
5. **Accuracy is the priority** — richer training objective (multi-task + contrastive
   auxiliary loss + fine-tuning) is in scope even though it is harder to train.
6. **Not in scope:** deep colony detection/segmentation, end-to-end single CNN that
   bypasses the hierarchy, cloud GPU training, contrastive-only training.

---

## 3. High-Level Architecture

```
                          ┌────────────────────────────────────────────┐
                          │              DEEP LEARNING (NEW)           │
                          │                                            │
   raw image (BGR) ─────► │  preprocess (resize 224/256, normalize)   │
                          │  pretrained backbone (EffNet-B0 / ResNet) │
                          │  ── fine-tuned on species (10 classes)    │
                          │  ── multi-task heads (group/gram/type)    │
                          │  ── contrastive auxiliary loss            │
                          │  global-pooled embedding (~1280-dim)      │
                          └───────────────┬────────────────────────────┘
                                          │  embedding vector
                                          ▼
                          ┌────────────────────────────────────────────┐
                          │        CLASSICAL LAYER (KEPT)             │
                          │  sklearn hierarchy, now trained on the     │
                          │  CNN embedding instead of 616-dim features │
                          │  organism_type → group → species → gram    │
                          └───────────────┬────────────────────────────┘
                                          ▼
                          ┌────────────────────────────────────────────┐
                          │        CLASSICAL CV (KEPT)                 │
                          │  colony detection (Otsu+contours)          │
                          │  colony shape model (7-dim features)       │
                          │  edge_density / laplacian_var (quality)    │
                          └───────────────┬────────────────────────────┘
                                          ▼
                          ┌────────────────────────────────────────────┐
                          │        INFERENCE ASSEMBLY (KEPT)           │
                          │  confidence blend, metadata overrides,     │
                          │  rejection gates, JSON basic/advanced,     │
                          │  PyQt5 GUI                                  │
                          └────────────────────────────────────────────┘
```

### Data flow (single image, inference)

1. `read_image()` → BGR ndarray.
2. **NEW** `extract_embedding(image)` → preprocess to `(3, H, W)`, forward through
   CNN, global-pool → embedding vector (float32, e.g. 1280-dim).
3. Sklearn `organism_type_model` / `group_model` / `group_species_models` /
   `gram_model` predict on a 1-row DataFrame built from the embedding columns.
4. Colony detection + shape model run unchanged (per-colony, 7-dim features).
5. Confidence blending + metadata overrides + rejection gates → JSON output.

---

## 4. Tech Stack

### Existing (kept)

| Component | Library |
| --------- | ------- |
| Colony detection / CV features | OpenCV (`opencv-python-headless==4.11.0.86`) |
| Hierarchical classifiers | scikit-learn 1.6.1 (RF / ExtraTrees / KNN-scaled) |
| Artifact serialization | joblib 1.4.2 |
| Data handling | numpy 2.2.6, pandas 2.2.3 |
| Desktop GUI | PyQt5 5.15.11 |
| Testing | pytest 8.3.5 |
| Runtime | Python 3.14.0 (project venv, Apple M2 / macOS) |

### NEW (to add)

| Component | Choice | Purpose |
| --------- | ------ | ------- |
| DL framework | **PyTorch** (+ `torchvision`) | CNN training, MPS/CPU acceleration |
| Pretrained backbone | `torchvision.models.efficientnet_b0` (primary), `resnet18` (fallback) | transfer-learning base; ImageNet weights |
| Embedding dim | EfficientNet-B0 global pool = **1280-dim** | replaces the 616-dim vector |
| Losses | CrossEntropy (species/group/gram/type heads) + **contrastive / triplet auxiliary** loss on embedding | supervised + metric-learning objective |
| Optimizer / scheduler | AdamW + cosine annealing (or ReduceLROnPlateau) | stable fine-tuning |
| Augmentation | torchvision/albumentations-style transforms: geometric (flip, rotate, scale-crop) **in addition to** existing photometric augmentation | generalization; modality robustness |
| Artifact format | `.pt` (PyTorch state_dict + meta) alongside existing `.joblib` | keeps joblib bundle for sklearn layer |

> **Dependency note:** PyTorch currently supports Python ≤3.13 for the most stable
> wheels; the project venv is Python 3.14.0. Plan: verify `pip install torch
> torchvision` on 3.14; if wheels are missing, create a **separate venv on Python
> 3.11/3.12/3.13** for training and embedding extraction. This does not change the
> design — the embedding model is framework-agnostic to the downstream sklearn layer.

---

## 5. Requirements & Resources

### Hardware

| Resource | Spec | Notes |
| -------- | ---- | ----- |
| CPU | Apple M2 (8-core) | training device fallback |
| Accelerator | Apple Metal (MPS) via PyTorch `mps` backend | preferred for fine-tuning |
| RAM | 16 GB | batch sizes must stay modest (e.g. 16–32) |
| Disk | ~41 GB free | dataset (~302 MB) + artifacts + torch cache |

### Software

- macOS (Darwin), zsh
- Python 3.14 venv at `.venv/` (existing); possibly secondary 3.11–3.13 venv for torch
- `git` (repo `BPA22_clone`, branch `main`)

### Data

| Resource | Location | Details |
| -------- | -------- | ------- |
| Training images | `Bacteria dataset/` | ~629 images, 10 organisms, 2 modalities (gram stain / media plate) |
| Metadata CSVs | `Bacteria dataset/dataset_full.csv`, `dataset_labels.csv`, `dataset_summary.csv` | labels + paths + imaging_type |
| Test images | `test_data/` | UI/manual verification images |
| Existing artifacts | `artifacts/bacteria_models.joblib` + `.metrics.json` | classical baseline; used for comparison |
| Pretrained weights | downloaded by torchvision (ImageNet) | requires network on first run |

### Taxonomy (fixed reference)

10 organisms across 4 groups (see `src/bacteria_assistant/config.py`). Labels:
organism_type (bacteria/fungi), gram_label (gram_positive/gram_negative/
non_bacterial_fungi), shape_label (cocci/bacilli/fungal), taxonomy_group (4 groups).

---

## 6. Training Design (NEW module: `src/bacteria_assistant/dl/`)

### 6.1 Directory layout (proposed)

```
src/bacteria_assistant/
├── config.py            # (extended: DL consts, paths, embedding columns)
├── features.py          # unchanged classical features
├── training.py          # classical training (unchanged baseline)
├── inference.py         # modified: image_vector <- embedding
├── ui/ (or scripts/)    # GUI entry unchanged
└── dl/
    ├── __init__.py
    ├── model.py         # backbone + multi-task heads + embedding layer
    ├── dataset.py       # ImageDataset (load, transform, label encode)
    ├── transforms.py    # train/val/inference transforms
    ├── losses.py        # combined loss (CE heads + contrastive auxiliary)
    ├── trainer.py       # training loop, MPS/CPU, early stopping, checkpointing
    ├── train_dl.py      # CLI entry: train embedding model
    └── extract.py       # extract_embedding(image) for training/inference
```

### 6.2 Model

```
inputs (3, H, W)
  → pretrained backbone (ImageNet weights)      [frozen for N epochs, then fine-tune]
  → global average pool → embedding z (1280-dim for EffNet-B0)
  → L2-normalize z (for contrastive)
  → heads (each a Linear / MLP):
      species_head  → 10 logits   (primary)
      group_head    → 4 logits    (multi-task regularizer)
      gram_head     → 3 logits    (multi-task regularizer)
      type_head     → 2 logits    (multi-task regularizer)
```

- **Embedding dim:** 1280 (EffNet-B0). If fine-tuning budget is tight, start frozen,
  train heads, then unfreeze last 2 blocks.
- **Why species-centric:** organism_type / gram / group are *deterministic functions*
  of the species (`ORGANISM_METADATA`), so one embedding trained primarily on species
  discrimination is sufficient for all image-level tasks. Multi-task heads still help
  as regularizers and for measuring per-head calibration.

### 6.3 Loss

```
L = L_species(CE) + w_g·L_group(CE) + w_g·L_gram(CE) + w_t·L_type(CE) + λ·L_contrastive(z)
```

- Contrastive term: supervised contrastive loss (pairs/triplets within species),
  or a simple normalized softmax/triplet on embeddings.
- `λ` and `w_*` tuned on the holdout; default `λ=0.1`, `w_*=0.3`.

### 6.4 Splits & leakage discipline (CRITICAL)

- Use **the same stratified 80/20 image-level holdout** as `training.py`
  (`train_test_split`, stratify on organism, `random_state=42`) so results are
  directly comparable to the classical baseline.
- **Augmentation applied to training fold only.**
- Report per-modality accuracy (gram-stain vs media-plate) — the known confounder
  from issue #8.
- Never tune hyperparameters on the test fold. Use a validation slice of the train
  fold (e.g. 20% of train as val) for early stopping/model selection.

### 6.5 Hyperparameters (initial)

| Param | Value |
| ----- | ----- |
| Input size | 224×224 (torchvision default) |
| Batch size | 16–32 (M2/16GB) |
| Epochs | 30–50 (early stop on val loss) |
| Optimizer | AdamW, lr 3e-4 (head), 1e-4 (unfrozen backbone) |
| Scheduler | cosine annealing |
| Weight decay | 1e-4 |
| Augmentation | random horizontal/vertical flip, ±10° rotation, scale-jitter crop, photometric jitter (reuse existing) |

---

## 7. Inference Integration

### 7.1 Artifact bundle

- Train → save `artifacts/embedding_model.pt` (state_dict + `embedding_dim`,
  `feature_version`, labels, transform config, model name).
- `train_models()` retrains the sklearn layer on embeddings → saves the existing
  `bacteria_models.joblib` but with:
  - `image_feature_columns` = `["emb_0" .. "emb_{d-1}"]`
  - `embedding_model_path` → `artifacts/embedding_model.pt`
  - `feature_version` bumped (e.g. 3)

### 7.2 `inference.py` change (minimal)

- `load_models()` also loads the embedding model if `embedding_model_path` is present.
- Replace:
  ```python
  image_features = extract_image_features(image)
  image_vector = pd.DataFrame([image_features])[artifacts["image_feature_columns"]]
  ```
  with:
  ```python
  emb = extract_embedding(image, embedding_model)   # np.ndarray (d,)
  image_vector = pd.DataFrame([emb])[artifacts["image_feature_columns"]]
  ```
- **Keep** `edge_density` and `laplacian_var` computation (cheap, used by the
  rejection gate). These can still come from `extract_image_features` or a small
  dedicated helper.
- Colony pipeline, shape model, blending, gates, JSON assembly: **unchanged**.

### 7.3 Backward compatibility

- If `embedding_model.pt` is absent, inference falls back to the classical 616-dim
  path so old artifacts still work (graceful degradation).

---

## 8. CLI / UI

- `train_model.py` gains a flag `--dl` to train the embedding + retrain sklearn layer.
- `predict_bacteria.py` unchanged (drives the same `predict_bacteria_image`).
- `bacteria_ui.py` unchanged (calls the same prediction entry point).
- New `src/bacteria_assistant/dl/train_dl.py` for CNN training standalone.

---

## 9. Testing Strategy

| Test | Scope |
| ---- | ----- |
| `tests/test_output_contract.py` | JSON basic/advanced/unknown schemas (must keep passing) |
| `tests/test_embedding.py` (new) | embedding shape, determinism, column alignment with sklearn |
| `tests/test_dl_training.py` (new, integration, marks slow) | tiny 2-epoch smoke train on a handful of images; asserts artifact exists + embedding dim |
| `tests/test_modality_metrics.py` (new) | per-modality accuracy reported |
| Manual | GUI: Analyze on `test_data/` images; compare basic/advanced JSON |

> CI note: dataset + artifacts are not committed to git; DL tests that need the dataset
> should be `pytest.mark.skipif` (like the existing contract tests) so the suite stays
> green without the dataset.

---

## 10. Known Risks & Mitigations

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| 629 images is small for deep learning | overfitting / weak generalization | transfer learning, heavy augmentation, early stopping, conservative capacity (EffNet-B0, not a huge ViT) |
| Python 3.14 torch wheel availability | blocked install | verify early; secondary 3.11–3.13 venv fallback |
| Modality confounder (gram vs media plate) | model keys on lighting not biology | per-modality metrics, cross-modality augmentation, frozen-backbone start |
| M2 training speed | slow iteration | frozen-backbone phase is fast; fine-tune selectively; track epoch times |
| Joblib bundle size grows (embedding sklearn) | disk | embedding dim 1280 × 4 models is small; joblib stays ~small |
| Accuracy ceiling from data, not method | target miss | accept realistic targets (see §1); design for easy data addition later |

---

## 11. Future / Follow-up (explicitly deferred)

- Deep colony detection (needs annotations) — out of scope.
- Pure contrastive-only training — out of scope (supervised signal too valuable).
- Cloud GPU training portability — out of scope, but trainer will keep device
  selection configurable for later.
- Test-time augmentation ensemble at inference — nice-to-have, only if targets unmet.
- Easy data-addition path (drop new images into dataset + re-run train) — design
  goal, achieved by keeping the training loop data-driven.

---

## 12. Reference: Baseline Metrics (classical artifact)

See `artifacts/bacteria_models.metrics.json`. Key holdout accuracies: organism_type
0.82, gram 0.64, group 0.45, species 0.33, shape 0.68.
