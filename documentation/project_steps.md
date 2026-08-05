# Project Build & Iteration Log (Presentation Notes)

This document lists the end‑to‑end steps taken to build the bacteria prediction system, including model training, output contracts, taxonomy fixes, UI creation, and accuracy improvements.

## 1) Workspace discovery & dataset validation
- Inspected project structure and confirmed dataset root: `data/dataset`.
- Validated CSV metadata (`dataset_full.csv`, `dataset_labels.csv`, `dataset_summary.csv`) to map image paths and organisms.
- Confirmed available images by class and imaging type (gram stain / media plate).

## 2) Baseline pipeline scaffold
- Created core Python package at `src/bacteria_assistant/` with modules:
  - `config.py` (label maps, model paths)
  - `features.py` (image + colony feature extraction)
  - `training.py` (model training + metrics)
  - `inference.py` (prediction + output contract)
- Added runners:
  - `scripts/train_model.py` (train + save artifacts)
  - `scripts/predict_bacteria.py` (predict on any image)
- Added `requirements.txt`, `pyproject.toml` for dependencies/pytest setup.
- Added `README.md` with usage and output examples.
- Added initial test: `tests/test_output_contract.py`.

## 3) Output contract implementation
- Implemented strict JSON outputs for:
  - **Basic output** (organism_type, total_colonies_detected, dominant_shape, confidence)
  - **Advanced output** (colonies list + final_morphology)
- Added colony geometry fields: area, perimeter, circularity, aspect_ratio, solidity, equivalent_diameter, mean_intensity.

## 4) Initial training & validation
- Trained baseline RandomForest classifiers for:
  - Gram label
  - Colony shape
- Saved artifacts to `artifacts/bacteria_models.joblib` and metrics JSON.
- Ran inference on sample images and confirmed JSON schema.

## 5) Bacteria name prediction
- Added bacteria‑name prediction model (organism classifier).
- Updated inference to return:
  - `predicted_bacteria_name`
  - `bacteria_type`
- Updated tests and README to include new fields.

## 6) Full taxonomy alignment (Gram+/Gram−/Fungi)
- Encoded dataset taxonomy based on your exact table:
  - Gram‑positive cocci, gram‑positive bacilli, gram‑negative bacilli.
  - Fungi (Candida albicans, Aspergillus niger).
- Added `organism_type_model` (bacteria vs fungi).
- Added `group_model` (taxonomy group prediction).
- Updated output to support fungi (`organism_type: fungi`, `bacteria_type: non_bacterial_fungi`, `dominant_shape: fungal`).

## 7) Constrained species prediction
- Added group‑constrained species prediction:
  - Predict taxonomy group first.
  - Predict species only within that group.
- Reduced cross‑group biological errors.

## 8) Accuracy improvements (model selection + features)
- Added spatial grayscale signature features to `features.py` for stronger discrimination.
- Added model selection per task (RandomForest vs ExtraTrees vs KNN+Scaling).
- Added group‑specific specialist species models to improve within‑group accuracy.
- Retrained and compared metrics after each improvement.

## 9) PyQt5 UI creation (retro Windows style)
- Built `scripts/bacteria_ui.py` with:
  - Upload image
  - Predict button
  - Summary output
  - “Show Details” toggle for full JSON
- Designed retro Windows UI (title bar, beveled frames, status bar, toolbar).
- Added robust image acceptance:
  - Validates upload using backend `read_image()`
  - Expanded supported formats (PNG/JPG/BMP/TIFF/WEBP)
  - Preview fallback if QPixmap fails

## 10) Reliability fixes & validation
- Fixed pandas chained‑assignment warnings.
- Ensured all outputs remain schema‑compliant after taxonomy expansion.
- Updated tests to allow fungi outputs.
- Re‑ran pytest after each change.

## 11) Final state summary
- Trained model artifact with:
  - Gram model
  - Organism type model (bacteria/fungi)
  - Taxonomy group model
  - Species model (global + group specialists)
  - Shape model (cocci/bacilli/fungal)
- UI supports upload + prediction + details.
- Output contracts validated by automated tests.

---

## Suggested presentation structure
1. **Problem & Dataset**: show dataset folders and CSV metadata.
2. **Pipeline Architecture**: features → models → JSON outputs.
3. **Taxonomy Handling**: Gram+/Gram−/Fungi mapping and constraints.
4. **UI Demo**: retro UI, upload, show prediction + details.
5. **Validation**: tests + sample outputs.
6. **Next Improvements**: deep learning model/transfer learning for stronger accuracy.


## final architecture
```md
[1] Input Image
      ↓
[2] Preprocessing
      ↓
[3] Global Feature Extraction   (image-level)
      ↓
[4] ML Hierarchy Inference
      ↓
      ├── organism_type
      ├── taxonomy_group
      ├── species_prediction
      └── gram_prediction

      ↓
[5] Colony Segmentation
      ↓
[6] Colony Feature Extraction
      ↓
[7] Morphology Classification (per colony)
      ↓
[8] Morphology Aggregation (image-level)
      ↓
[9] CONSISTENCY CHECK (NEW — inside current system)
      ↓
[10] Output Builder
```
##  Global Feature Extraction   (image-level)
Global Feature Extraction means everything we compute from the entire image (not individual colonies) to help the model recognize species and gram type. Global Feature Extraction = converting the WHOLE IMAGE into a numerical vector that represents color, texture, structure, and overall patterns