# Morphology & Bacteria Prediction Pipeline — Full System Architecture (Detailed)

This document provides a **deep, step‑by‑step explanation** of how morphology and bacteria prediction are performed, covering all preprocessing, feature extraction, training, validation, evaluation, and the full end‑to‑end architecture.

---

## 1) System Goals, Inputs, and Outputs

### 1.1 Primary input
- Microscopic image (gram‑stain or plate), provided via UI or CLI.

### 1.2 Outputs (contract)

**Basic output**
```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies_detected": 42,
  "dominant_shape": "bacilli",
  "confidence": 0.91
}
```

**Advanced output**
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

## 2) Dataset & Taxonomy Definition

### 2.1 Dataset location
- Root: `data/dataset/`
- CSVs used:
  - `dataset_full.csv` (image path + organism + imaging type)
  - `dataset_labels.csv` (class metadata)
  - `dataset_summary.csv` (counts)

### 2.2 Organism taxonomy (explicit biological mapping)
- **Gram‑positive cocci**: *Staphylococcus aureus*, *Streptococcus pyogenes*, *Enterococcus faecalis*
- **Gram‑positive bacilli**: *Bacillus subtilis*, *Clostridium sporogenes*
- **Gram‑negative bacilli**: *Escherichia coli*, *Klebsiella pneumoniae*, *Pseudomonas aeruginosa*
- **Fungi**: *Candida albicans*, *Aspergillus niger*

This mapping drives:
- `organism_type` (bacteria | fungi)
- `gram_label` (gram_positive | gram_negative | non_bacterial_fungi)
- `shape_label` (cocci | bacilli | fungal)
- `taxonomy_group`

## 3) Preprocessing & Morphology Feature Pipeline

### 3.1 CSV ingestion and normalization
- Read `dataset_full.csv` with pandas
- Normalize organism labels (underscore removal + whitespace trimming)
- Normalize imaging type (`gram stain` / `media plate`)

### 3.2 Image loading & sanitization
- `read_image()` uses `cv2.imdecode` to safely read paths with spaces/unicode
- Fails fast if the image is unreadable

### 3.3 Global image feature extraction (species‑level signal)
Executed in `features.py`:

**Color statistics**
- RGB mean and std
- HSV mean and std
- Gray mean and std

**Texture / sharpness**
- Laplacian variance
- Edge density (Canny)

**Color histograms**
- 8‑bin RGB histograms

**Spatial morphology signature**
- 24×24 downsampled gray map
- Encodes coarse colony texture/layout

### 3.4 Colony segmentation (morphology pipeline)
1. Gray conversion
2. Gaussian blur
3. Otsu threshold (binary + inverted)
4. Morphological open/close
5. Contour filtering by area

### 3.5 Colony morphology features (per colony)
- Area, perimeter
- Circularity $4\pi A / P^2$
- Aspect ratio
- Solidity (area / hull area)
- Equivalent diameter
- Mean intensity

These feed the **shape model** and the advanced JSON output.

---

## 4) Training Pipeline (Hierarchical)

### 4.1 Training orchestration
Implemented in `src/bacteria_assistant/training.py` using stratified train/test splits.
All models share the same **image feature vector** unless otherwise stated.

### 4.2 Hierarchical model stack (image‑level)

#### A) Organism Type Model
- Task: `bacteria` vs `fungi`
- Purpose: block impossible gram predictions for fungi

#### B) Taxonomy Group Model
- Task: `gram_positive_cocci`, `gram_positive_bacilli`, `gram_negative_bacilli`, `fungi`
- Purpose: constrain species prediction to biologically valid groups

#### C) Species Model (Global)
- Task: 10 organism classes
- Used as fallback if a group specialist fails

#### D) Group‑Specialist Species Models
- One per taxonomy group
- Reduces cross‑group misclassification (e.g., gram+ cocci vs gram− bacilli)

#### E) Gram Model
- Task: gram_positive vs gram_negative
- Trained only on bacteria rows

### 4.3 Morphology model (colony‑level)
- Task: cocci / bacilli / fungal
- Trained on colony geometry features

### 4.4 Model selection strategy
For each model, the system evaluates **three candidate classifiers**:
- RandomForest
- ExtraTrees
- KNN + StandardScaler

The candidate with the highest validation accuracy is selected and stored.

---

## 5) Validation & Evaluation (Detailed)

### 5.1 Metrics recorded per model
For each classifier:
- Accuracy
- Precision / Recall / F1 for each class
- Macro and weighted averages

Saved in:
```
artifacts/bacteria_models.metrics.json
```

### 5.2 Output contract tests
`tests/test_output_contract.py` verifies:
- Required fields exist
- Valid organism types and shapes
- Confidence bounds

### 5.3 Sanity inference checks
Sample images are run through inference to validate:
- output structure
- reasonable colony counts
- morphology aggregation

---

## 6) Inference Pipeline (Detailed)

Implemented in `src/bacteria_assistant/inference.py`.

### 6.1 Hierarchical flow
1. Load image → extract global features
2. **Organism type model** (bacteria vs fungi)
3. **Taxonomy group model** (cocci/bacilli/fungi group)
4. **Group‑specific species model**
5. Gram model (bacteria only)
6. Colony detection → colony geometry features
7. Shape model → per‑colony morphology
8. Aggregation → dominant shape + distribution
9. JSON output assembly

### 6.2 Morphology aggregation logic
- Dominant shape is the most frequent colony class
- Confidence blends model probability + dominance ratio
- Distribution determined by nearest‑neighbor spacing among colonies

---

## 7) Full System Architecture (Detailed)

```
┌────────────────────────────────────────────────────────┐
│                    UI / CLI Layer                      │
│           (PyQt5 UI, scripts/*.py runners)             │
└──────────────────────────┬─────────────────────────────┘
                  │ Image Upload / Path
                  ▼
┌────────────────────────────────────────────────────────┐
│                Inference Orchestrator                  │
│              predict_bacteria_image()                  │
└──────────────────────────┬─────────────────────────────┘
                  │
                  ├──► Image Feature Extractor
                  │       (color + texture + spatial)
                  │
                  ├──► Organism Type Model
                  │       (bacteria vs fungi)
                  │
                  ├──► Taxonomy Group Model
                  │       (gram+ cocci / gram+ bacilli / gram- bacilli / fungi)
                  │
                  ├──► Group‑Specific Species Model
                  │       (constrained species prediction)
                  │
                  ├──► Gram Model (bacteria only)
                  │
                  ├──► Colony Segmentation
                  │
                  ├──► Colony Morphology Model
                  │       (cocci / bacilli / fungal)
                  │
                  └──► JSON Output Builder
                        (basic + advanced)
```

---

## 8) UI Layer (PyQt5)

**File:** `scripts/bacteria_ui.py`

Features:
- Upload image
- Predict and show results
- Detailed JSON toggle (Show/Hide)
- Retro Windows UI styling
- Robust image acceptance (fallback if preview fails)

---

## 9) Artifacts & Outputs

- `artifacts/bacteria_models.joblib` → trained model bundle
- `artifacts/bacteria_models.metrics.json` → evaluation metrics
- JSON response returned to UI/CLI

---

## 10) Dependency Stack

Core libraries:
- `opencv-python-headless`
- `numpy`, `pandas`
- `scikit-learn`
- `joblib`
- `PyQt5`

---

## 11) Future Improvements

- Add CNN / transfer learning feature extractor
- Improve fungi detection with more samples
- Add confidence threshold & “uncertain” label
- Add model monitoring dashboard

---

## Appendix: Key File Map

- `src/bacteria_assistant/config.py` → taxonomy + constants
- `src/bacteria_assistant/features.py` → preprocessing + colony features
- `src/bacteria_assistant/training.py` → training pipeline
- `src/bacteria_assistant/inference.py` → inference pipeline
- `scripts/bacteria_ui.py` → UI layer
- `scripts/train_model.py` / `scripts/predict_bacteria.py` → CLI runners
- `tests/test_output_contract.py` → output validation
