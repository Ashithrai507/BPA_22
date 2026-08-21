# BPA-22 — Bacteria Lab Assistant Predictor

A classical machine-learning pipeline that analyzes microscopy images of bacterial and
fungal cultures and produces structured lab-grade predictions: **organism type,
taxonomy group, species, gram stain**, and **colony morphology** — via a hierarchical
`scikit-learn` classifier stack with a desktop GUI.

```
Input image ──► global features ──► organism_type ──► taxonomy_group ──► species ──► gram
                                 └──────────► colony segmentation ──► shape ──► aggregation ──► JSON
```
---

## Features

- **Hierarchical ML inference** — each classifier narrows the decision space, reducing
  biologically invalid predictions.
- **Organism classification** — `bacteria` vs `fungi` across **10 supported species**.
- **Gram classification** — `gram_positive` / `gram_negative` (bacteria only; fungi
  mapped to `non_bacterial_fungi`).
- **Group-constrained species prediction** — species models are trained per taxonomy
  group; the global model is a fallback.
- **Colony detection & measurement** — area, perimeter, circularity, aspect ratio,
  solidity, equivalent diameter, mean intensity.
- **Morphology prediction** — `cocci`, `bacilli`, `spiral` (geometry heuristic),
  `fungal` (override).
- **Structured JSON output** — exact contracts in **basic** and **advanced** mode.
- **Confidence gating** — low-quality or out-of-distribution images are flagged as
  `unknown` instead of returning a false positive.
- **Desktop GUI** — retro Win95-styled PyQt5 application with live image preview and
  raw JSON inspection.

---

## Supported Taxonomy

| Taxonomy group       | Organisms                                                          |
| -------------------- | ------------------------------------------------------------------ |
| Gram-positive cocci  | *Staphylococcus aureus*, *Streptococcus pyogenes*, *Enterococcus faecalis* |
| Gram-positive bacilli| *Bacillus subtilis*, *Clostridium sporogenes*                      |
| Gram-negative bacilli| *Escherichia coli*, *Klebsiella pneumoniae*, *Pseudomonas aeruginosa* |
| Fungi                | *Candida albicans*, *Aspergillus niger*                            |

---

## Installation

Requires **Python 3.11+**.

```bash
python -m pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`, `opencv-python-headless`,
`joblib`, `PyQt5`, `pytest`.

---

## Usage

### 1. Train the model bundle

```bash
make train
# or
python scripts/train_model.py
```

Artifacts written to:

- `artifacts/bacteria_models.joblib` — serialized model bundle
- `artifacts/bacteria_models.metrics.json` — per-model evaluation metrics, including
  `per_modality_metrics` (gram-stain vs media-plate held-out accuracy) and
  `training_meta.feature_version`

> The bundled artifact was produced in a different workspace. Retrain locally to
> refresh feature paths and metrics.

### 2. Run prediction from the CLI

```bash
# Basic output
python scripts/predict_bacteria.py \
  --image "data/dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png" \
  --mode basic

# Advanced output (per-colony measurements + morphology summary)
python scripts/predict_bacteria.py \
  --image "data/dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png" \
  --mode advanced
```

### 3. Launch the desktop UI

```bash
python scripts/bacteria_ui.py
```

- Open an image (PNG / JPG / BMP / TIFF / WEBP)
- Click **Analyze** to run prediction
- Toggle **Show Details** to view the full JSON payload

### 4. Run the test suite

```bash
python -m pytest -q
```

Validates the basic/advanced JSON output contracts. Tests skip (with a visible
reason) if the dataset or trained model is missing — they never pass vacuously.

---

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow.

```bash
make setup        # venv + runtime + dev deps
make train        # rebuild the model bundle (needs data/dataset present)
make test         # run tests
make lint         # ruff check + format check
make format       # auto-format with ruff
make run-ui       # launch the desktop UI
make help         # list all targets
```

**Dataset & artifact are not in git.** Copy the dataset (~302 MB) into
`data/dataset` and run `make train`, or copy an existing
`artifacts/bacteria_models.joblib` (~662 MB). Per-developer paths can be
overridden via `.env` (see `.env.example`).

CI (GitHub Actions) runs lint + full tests on every PR. Because CI has no dataset,
it uses a seeded model artifact (Actions cache, falling back to the `model-artifact`
GitHub release). After retraining, reseed with `make ci-upload-artifact`.

`main` is protected: pull requests require an approving review and green CI.

---

## Output Contracts

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

`distribution` values: `clustered`, `mixed`, `dispersed`, `isolated`.

### Uncertain / rejected input

When the image is unreadable, contains no valid colonies, or fails confidence/quality
gates, all label fields are returned as `unknown` (colony counts are still reported).

---

## How It Works

### Feature extraction

- **Global image features (616-dim)** — per image: RGB/HSV/grayscale mean & std,
  Laplacian variance (sharpness), Canny edge density, 8-bin RGB color histograms, and a
  24×24 spatial grayscale signature encoding colony layout. The grayscale is
  **CLAHE-normalized** (`clipLimit=2.0`) before the spatial/edge/Laplacian features are
  computed to remove the gram-stain vs media-plate illumination confounder.
- **Photometric augmentation** — training images are augmented with brightness/contrast
  jitter and CLAHE-clip variants (train fold only) so models stop keying on lighting.
- **Colony features (7-dim)** — per segmented colony: area, perimeter, circularity
  (`4πA/P²`), aspect ratio, solidity, equivalent diameter, mean intensity.

### Hierarchical model stack

| Model              | Task                                                        |
| ------------------ | ----------------------------------------------------------- |
| `organism_type_model` | bacteria vs fungi                                         |
| `group_model`      | taxonomy group (4 classes)                                  |
| `organism_model`   | global species classifier (10 classes, fallback)            |
| `group_species_models` | per-group specialist species classifiers                |
| `gram_model`       | gram_positive vs gram_negative (bacteria rows only)         |
| `shape_model`      | colony morphology: cocci / bacilli / fungal                 |

Every classifier is auto-selected between **RandomForest**, **ExtraTrees**, and
**KNN + StandardScaler** based on hold-out validation accuracy.

### Inference flow

1. Extract global features from the image.
2. Predict organism type → taxonomy group → group-constrained species.
3. Predict gram label; apply **metadata consistency overrides** so the output can never
   contradict the species' known biology.
4. Segment colonies; predict a shape per colony (with a spiral geometry heuristic).
5. Aggregate to dominant shape + spatial distribution + blended confidence.
6. Gate the result through the confidence/quality checks; build the JSON output.

---

## Project Structure

```
BPA_22/
├── scripts/                       # CLI entry points
│   ├── bacteria_ui.py             # PyQt5 desktop UI
│   ├── train_model.py             # train + save artifacts
│   ├── predict_bacteria.py        # predict on a single image
│   └── run_protein_pipeline.py    # Phase 2: species -> protein sequence
├── src/
│   ├── bacteria_assistant/        # Phase 1: CV/ML pipeline
│   │   ├── config.py              # taxonomy, thresholds, paths, feature version
│   │   ├── features.py            # image + colony feature extraction, CLAHE, augmentation
│   │   ├── training.py            # hierarchical training pipeline
│   │   └── inference.py           # prediction + output assembly
│   └── protein_engine/            # Phase 2: protein retrieval engine
│       ├── taxonomy/              # species name + NCBI taxonomy ID
│       ├── retrieval/             # UniProt / NCBI / PDB clients + cache
│       ├── ranking/               # essential / virulence / resistance scoring
│       ├── sequence/              # validation + FASTA/JSON export
│       └── documentation/         # design & architecture specs
├── morphology/                    # standalone prototype pipeline (research)
├── tests/
│   ├── test_output_contract.py    # JSON output contract tests
│   └── test_modality_fix.py       # modality-confounder unit tests
├── data/                          # dataset + test images (not in git)
├── reference_data/                # curated ranking DBs (not in git)
├── output/                        # pipeline runtime outputs (not in git)
├── artifacts/                     # trained model bundle + metrics (not in git)

├── documentation/                 # architecture & workflow deep-dives
├── Makefile                       # dev workflow targets (setup/train/test/lint/...)
├── pyproject.toml                 # packaging, pytest, ruff config
├── requirements.txt               # pinned runtime dependencies
├── .env.example                   # optional per-dev path overrides
├── CONTRIBUTING.md                # contributor workflow
├── .pre-commit-config.yaml        # optional git hooks
└── .github/
    ├── workflows/ci.yml           # lint + tests on every PR
    └── PULL_REQUEST_TEMPLATE.md
```

---

## Documentation

- [`documentation/classical_ml_workflow(consider).md`](<documentation/classical_ml_workflow(consider).md>) —
  complete end-to-end walkthrough of the ML pipeline.
- [`documentation/pipeline_architecture.md`](documentation/pipeline_architecture.md) —
  detailed system architecture.
- [`documentation/follow.md`](documentation/follow.md) — feature/spec notes.
- [`src/protein_engine/documentation/design.md`](src/protein_engine/documentation/design.md) —
  Phase 2 protein engine system design.
- [`src/protein_engine/documentation/architecture.md`](src/protein_engine/documentation/architecture.md) —
  Phase 2 protein engine technical architecture.
- [`docs/superpowers/specs/2026-08-06-protein-engine-design.md`](docs/superpowers/specs/2026-08-06-protein-engine-design.md) —
  Phase 2 protein engine design spec (pending review).

---

## Known Limitations

- **Moderate accuracy on hard splits** — e.g. species-level accuracy is limited by
  small per-class sample sizes and strong visual similarity between species.
- **Spiral is heuristic-only** — no spiral training samples exist; the class is inferred
  from geometry (`aspect_ratio ≥ 3.0`, `circularity ≤ 0.35`).
- **Two imaging modalities** — gram-stain and media-plate images differ visually; the
  models must generalize across both.
- **Rejection thresholds** are conservative; some real but noisy images may be flagged
  as `unknown`.

## Future Improvements

- CNN / transfer-learning feature extractor for stronger discrimination.
- Larger, balanced dataset with explicit spiral samples.
- Confidence-threshold calibration and an explicit `uncertain` tier.
- Model monitoring / drift dashboard.
