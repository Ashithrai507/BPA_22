# BPA-22 — Bacteria Lab Assistant Predictor

A machine-learning pipeline that analyzes microscopy images of bacterial and
fungal cultures and produces structured lab-grade predictions: **organism type,
taxonomy group, species, gram stain**, and **colony morphology** — via a hierarchical
`scikit-learn` classifier stack (optionally powered by deep-learning embeddings).

Ships with a desktop GUI, a **web app** (FastAPI REST backend + React SPA with a
SQLite prediction history), and a **protein ranking engine** that retrieves and
scores folding-ready proteins for the predicted species.

```
Input image ──► image features ──► organism_type ──► taxonomy_group ──► species ──► gram
                                 └──────────► colony segmentation ──► shape ──► aggregation ──► JSON
```

Image features come either from hand-engineered CV descriptors (default) or from an
**optional deep-learning embedding model** that replaces them with learned features.

---

## Features

- **Hierarchical ML inference** — each classifier narrows the decision space, reducing
  biologically invalid predictions.
- **Optional DL embeddings** — a pretrained `efficientnet_b0` multi-task encoder
  (ResNet / MobileNet backbones also available) can replace the hand-engineered image
  features; species inference then runs with test-time augmentation, group-constrained.
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
- **Web app** — FastAPI REST backend + React (Vite + Tailwind) SPA: drag-and-drop image
  analysis, per-colony results table, persistent prediction history with delete
  (`data/bpa.db`), and interactive protein ranking.
- **Protein ranking engine** — for a predicted (or named) species: resolves the NCBI
  taxonomy ID, retrieves candidate proteins from UniProt/PDB, scores them
  (essentiality / virulence / resistance), and exports FASTA. Supports batch mode and
  offline reuse of the retrieval cache.

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
`joblib`, `torch`, `torchvision`, `requests`, `PyQt5`, `pytest`.

---

## Usage

### 1. Train the model bundle

```bash
make train
# or
python scripts/train_model.py
```

Optional deep-learning track — train an embedding model first, then retrain the
classical hierarchy on top of the learned embeddings:

```bash
python scripts/train_dl.py                       # writes artifacts/embedding_model.pt
python scripts/train_model.py --dl artifacts/embedding_model.pt
```

Artifacts written to:

- `artifacts/bacteria_models.joblib` — serialized model bundle
- `artifacts/bacteria_models.metrics.json` — per-model evaluation metrics, including
  `per_modality_metrics` (gram-stain vs media-plate held-out accuracy) and
  `training_meta.feature_version`
- `artifacts/embedding_model.pt` — DL embedding checkpoint (only when the DL track is used)

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

### 4. Evaluate model quality

```bash
make evaluate
# or
python scripts/evaluate_models.py
```

Prints a precision / recall / F1 report per model from the trained artifact's metrics
JSON. To measure end-to-end DL-assisted inference accuracy on the shared 80/20 holdout:

```bash
python scripts/evaluate_dl_holdout.py
```

### 5. Rank proteins for a species (CLI)

```bash
# From a petri-dish image (predicts the species, then ranks its proteins)
python scripts/run_protein_pipeline.py --image path/to/plate.png --top-n 10

# Or directly from a species name
python scripts/run_protein_pipeline.py --species "Bacillus subtilis" --top-n 10

# Reuse the retrieval cache only (no network calls)
python scripts/run_protein_pipeline.py --species "Escherichia coli" --offline
```

Writes ranked proteins with scores plus FASTA/JSON exports to `output/` by default.

### 6. Run the web app

Requires **Node 18+** for the frontend and `fastapi`/`uvicorn` for the backend.

```bash
# 1. Backend — FastAPI on http://localhost:8000
python -m pip install -r server/requirements.txt
uvicorn server.main:app --port 8000

# 2. Frontend — Vite dev server on http://localhost:5173 (proxies /api to :8000)
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>:

- **Image Analysis** — drag & drop a plate image, inspect the per-colony results table
  and raw JSON; every run is stored in the history (`data/bpa.db`, created at runtime).
- **Protein Ranking** — pick one of the 10 supported species (or paste several for
  batch mode), rank folding-ready candidates, download FASTA.
- **History** — browse past predictions across pages, see protein-analysis status,
  open or delete entries.

REST endpoints: `POST /api/predict`, `GET /api/predict/history`,
`GET|DELETE /api/predict/{id}`, `GET /api/species`,
`GET /api/proteins/history`, `POST /api/proteins`, `POST /api/proteins/batch`.

### 7. Run the test suite

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
make evaluate     # print the per-model quality report from the metrics JSON
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

### Deep-learning embedding track (optional)

Instead of the hand-engineered global features, a pretrained CNN encoder can produce
the image representation:

- **Backbone** — `efficientnet_b0` by default (1280-d pooled embedding); `resnet18`,
  `resnet50`, and `mobilenet_v3_large` are also supported.
- **Multi-task heads** — species / group / gram / organism-type heads regularize
  training and supply the species probabilities at inference; downstream sklearn
  classifiers consume the pooled embedding.
- **Training** — backbone frozen for the first epochs, then fine-tuned end-to-end
  with auxiliary + contrastive losses (`scripts/train_dl.py`).
- **Inference** — when the artifact references an embedding checkpoint, `load_models()`
  attaches it automatically; species prediction uses test-time augmentation restricted
  to the predicted taxonomy group. The image-quality rejection gate still runs on the
  classical edge/sharpness statistics.

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

1. Extract image features (classical descriptors, or a DL embedding when the
   artifact was trained with `--dl`).
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
│   ├── train_model.py             # train + save artifacts (--dl for embedding retrain)
│   ├── train_dl.py                # train the DL embedding model
│   ├── evaluate_models.py         # per-model quality report from metrics JSON
│   ├── evaluate_dl_holdout.py     # DL-assisted inference accuracy on the holdout
│   ├── predict_bacteria.py        # predict on a single image
│   └── run_protein_pipeline.py    # Phase 2: species -> protein sequence
├── src/
│   ├── bacteria_assistant/        # Phase 1: CV/ML pipeline
│   │   ├── config.py              # taxonomy, thresholds, paths, feature version
│   │   ├── features.py            # image + colony feature extraction, CLAHE, augmentation
│   │   ├── training.py            # hierarchical training pipeline
│   │   ├── inference.py           # prediction + output assembly
│   │   └── dl/                    # optional DL embedding track (torch)
│   │       ├── model.py           # pretrained backbone + multi-task heads
│   │       ├── dataset.py         # image dataset + label encoders
│   │       ├── transforms.py      # train/eval augmentations
│   │       ├── losses.py          # auxiliary + contrastive losses
│   │       ├── trainer.py         # fit_embedding_model, device helpers
│   │       └── extract.py         # checkpoint loading, embeddings, TTA species probs
│   └── protein_engine/            # Phase 2: protein retrieval engine
│       ├── taxonomy/              # species name + NCBI taxonomy ID
│       ├── retrieval/             # UniProt / NCBI / PDB clients + cache
│       ├── ranking/               # essential / virulence / resistance scoring
│       ├── sequence/              # validation + FASTA/JSON export
│       └── documentation/         # design & architecture specs
├── server/                        # Phase 3: FastAPI REST backend for the web app
│   ├── main.py                    # app assembly, CORS, routers
│   ├── db.py                      # SQLite audit log (predictions + protein analyses)
│   ├── models.py                  # Pydantic request/response schemas
│   ├── dependencies.py            # cached model loading + DB path
│   └── routers/
│       ├── predict.py             # POST /api/predict, history/detail/delete
│       └── proteins.py            # species list, protein ranking (+ batch/history)
├── frontend/                      # React 19 + Vite + Tailwind SPA
│   └── src/
│       ├── pages/                 # PredictPage, ProteinPage, HistoryPage
│       ├── components/            # upload, result tables, FASTA export
│       └── api.js                 # axios client (baseURL /api)
├── morphology/                    # standalone prototype pipeline (research)
├── tests/                         # contract tests + classical/DL unit suites
│   ├── test_output_contract.py    # JSON output contract tests
│   ├── test_modality_fix.py       # modality-confounder unit tests
│   └── test_dl_*.py, ...          # DL dataset/model/training/inference suites
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

- [`documentation/model_metrics.md`](documentation/model_metrics.md) — full holdout
  metrics for the DL hybrid vs classical baseline (accuracy, per-class reports, insights).
- [`documentation/bacteria_assistant_modules.md`](documentation/bacteria_assistant_modules.md) —
  function-by-function reference for every `bacteria_assistant/` module (classical + DL).
- [`documentation/image_processing_pipeline.md`](documentation/image_processing_pipeline.md) —
  step-by-step walkthrough of what happens between input image and JSON output.
- [`documentation/developer.md`](documentation/developer.md) — end-to-end developer
  guide: building the system from scratch.
- [`documentation/project_steps.md`](documentation/project_steps.md) — build &
  iteration log (training, contracts, accuracy work).
- [`documentation/questions.md`](documentation/questions.md) — architecture concept Q&A.
- [`src/protein_engine/documentation/design.md`](src/protein_engine/documentation/design.md) —
  Phase 2 protein engine system design.
- [`src/protein_engine/documentation/architecture.md`](src/protein_engine/documentation/architecture.md) —
  Phase 2 protein engine technical architecture.
- [`docs/system_design_architecture.md`](docs/system_design_architecture.md) and
  [`docs/step_log.md`](docs/step_log.md) — deep-learning redesign design draft and step log.
- [`docs/superpowers/specs/2026-08-13-dl-integration-design.md`](docs/superpowers/specs/2026-08-13-dl-integration-design.md) —
  DL embedding integration spec (merged).
- [`docs/superpowers/specs/2026-08-06-protein-engine-design.md`](docs/superpowers/specs/2026-08-06-protein-engine-design.md) —
  Phase 2 protein engine design spec (merged).
- [`docs/superpowers/specs/2026-08-17-web-frontend-design.md`](docs/superpowers/specs/2026-08-17-web-frontend-design.md) —
  web frontend + FastAPI backend design spec.

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

- Larger, balanced dataset with explicit spiral samples.
- Confidence-threshold calibration and an explicit `uncertain` tier.
- Model monitoring / drift dashboard.
