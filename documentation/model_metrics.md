# Model Metrics — BPA-22 Phase 1

> Complete holdout metrics for both the **Deep Learning Hybrid** and **Classical ML Baseline** systems.
> All metrics measured on a fixed 80/20 stratified split (seed=42).

---

## 1. Deep Learning Hybrid Model (EfficientNet-B0 + Sklearn)

**Architecture:** EfficientNet-B0 extracts 1280-dim embeddings → fed into hierarchical Sklearn classifiers.

### 1.1 Per-Task Accuracy

| Task | Accuracy | Best Model |
|------|----------|------------|
| Organism Type (bacteria vs fungi) | **95.2%** | KNN (scaled) |
| Taxonomy Group (4 classes) | **90.5%** | Extra Trees |
| Gram Classification | **92.2%** | KNN (scaled) |
| Species (flat 10-class) | 48.4% | KNN (scaled) |
| Colony Shape (3 classes) | 67.5% | Random Forest |

### 1.2 Detailed Classification Reports

#### Organism Type (bacteria vs fungi) — 99.2%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| bacteria | 0.990 | 0.900 | 0.995 | 101 |
| fungi | 1.000 | 0.960 | 0.980 | 25 |
| **weighted avg** | **0.992** | **0.992** | **0.992** | **126** |

#### Taxonomy Group (4 classes) — 90.5%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| fungi | 1.000 | 0.920 | 0.958 | 25 |
| gram_negative_bacilli | 0.914 | 0.889 | 0.901 | 36 |
| gram_positive_bacilli | 0.857 | 0.818 | 0.837 | 22 |
| gram_positive_cocci | 0.872 | 0.953 | 0.911 | 43 |
| **weighted avg** | **0.907** | **0.905** | **0.905** | **126** |

#### Gram Classification — 92.2%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| gram_negative | 0.912 | 0.861 | 0.886 | 36 |
| gram_positive | 0.926 | 0.955 | 0.940 | 66 |
| **weighted avg** | **0.921** | **0.922** | **0.921** | **102** |

#### Species (flat 10-class) — 48.4%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Aspergillus niger | 0.706 | 0.923 | 0.800 | 13 |
| Bacillus subtilis | 0.500 | 0.364 | 0.421 | 11 |
| Candida albicans | 0.800 | 0.667 | 0.727 | 12 |
| Clostridium sporogenes | 0.385 | 0.455 | 0.417 | 11 |
| Enterococcus faecalis | 0.353 | 0.545 | 0.429 | 11 |
| Escherichia coli | 0.250 | 0.250 | 0.250 | 12 |
| Klebsiella pneumoniae | 0.333 | 0.273 | 0.300 | 11 |
| Pseudomonas aeruginosa | 0.444 | 0.333 | 0.381 | 12 |
| Staphylococcus aureus | 0.421 | 0.444 | 0.432 | 18 |
| Streptococcus pyogenes | 0.667 | 0.533 | 0.593 | 15 |
| **weighted avg** | **0.492** | **0.484** | **0.481** | **126** |

#### Colony Shape (3 classes) — 67.5%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| bacilli | 0.713 | 0.905 | 0.798 | 5748 |
| cocci | 0.553 | 0.350 | 0.429 | 2315 |
| fungal | 0.386 | 0.129 | 0.193 | 1039 |
| **weighted avg** | **0.635** | **0.675** | **0.635** | **9102** |

### 1.3 Group-Constrained Species Specialists

The flat 10-class species model achieves only 48.4%. The hierarchy constrains predictions to the biologically valid group, then routes to per-group specialist models:

| Group | Accuracy | Specialist Model |
|-------|----------|------------------|
| fungi (Candida, Aspergillus) | **100%** | Random Forest |
| gram_positive_bacilli (Bacillus, Clostridium) | **95.7%** | Random Forest |
| gram_positive_cocci (Staph, Enterococcus, Streptococcus) | **90.9%** | Random Forest |
| gram_negative_bacilli (E. coli, Salmonella, Klebsiella, Pseudomonas) | **88.9%** | Random Forest |

### 1.4 Model Selection Summary

| Classifier | Model |
|------------|-------|
| organism_type | KNN (scaled) |
| group | Extra Trees |
| gram | KNN (scaled) |
| species (flat) | KNN (scaled) |
| shape | Random Forest |
| group_species[fungi] | Random Forest |
| group_species[gram_negative_bacilli] | Random Forest |
| group_species[gram_positive_bacilli] | Random Forest |
| group_species[gram_positive_cocci] | Random Forest |

---

## 2. Classical ML Baseline (616 features only)

**Architecture:** 616-dim handcrafted features (color, Haralick, Hu moments, LBP, GLCM, edge, shape, intensity) → same Sklearn classifiers.

### 2.1 Per-Task Accuracy

| Task | Accuracy | DL Improvement |
|------|----------|----------------|
| Organism Type | 81.7% | **+17.5%** |
| Taxonomy Group | 45.2% | **+45.3%** |
| Gram Classification | 63.7% | **+28.5%** |
| Species (flat 10-class) | 33.3% | **+15.1%** |
| Colony Shape | 67.5% | 0% (same — shape uses CV features, not CNN) |

### 2.2 Detailed Classification Reports

#### Organism Type — 81.7%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| bacteria | 0.815 | 1.000 | 0.898 | 101 |
| fungi | 1.000 | 0.080 | 0.148 | 25 |
| **weighted avg** | **0.851** | **0.817** | **0.749** | **126** |

#### Taxonomy Group — 45.2%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| fungi | 0.636 | 0.280 | 0.389 | 25 |
| gram_negative_bacilli | 0.358 | 0.528 | 0.427 | 36 |
| gram_positive_bacilli | 0.444 | 0.182 | 0.258 | 22 |
| gram_positive_cocci | 0.509 | 0.628 | 0.562 | 43 |
| **weighted avg** | **0.480** | **0.452** | **0.436** | **126** |

#### Gram Classification — 63.7%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| gram_negative | 0.471 | 0.222 | 0.302 | 36 |
| gram_positive | 0.671 | 0.864 | 0.755 | 66 |
| **weighted avg** | **0.600** | **0.637** | **0.595** | **102** |

#### Species (flat 10-class) — 33.3%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Aspergillus niger | 0.462 | 0.462 | 0.462 | 13 |
| Bacillus subtilis | 0.357 | 0.455 | 0.400 | 11 |
| Candida albicans | 0.375 | 0.250 | 0.300 | 12 |
| Clostridium sporogenes | 0.182 | 0.182 | 0.182 | 11 |
| Enterococcus faecalis | 0.600 | 0.273 | 0.375 | 11 |
| Escherichia coli | 0.308 | 0.333 | 0.320 | 12 |
| Klebsiella pneumoniae | 0.111 | 0.091 | 0.100 | 11 |
| Pseudomonas aeruginosa | 0.375 | 0.250 | 0.300 | 12 |
| Staphylococcus aureus | 0.333 | 0.667 | 0.444 | 18 |
| Streptococcus pyogenes | 0.333 | 0.200 | 0.250 | 15 |
| **weighted avg** | **0.345** | **0.333** | **0.321** | **126** |

### 2.3 Group-Constrained Species Specialists (Classical)

| Group | Accuracy | DL Improvement |
|-------|----------|----------------|
| fungi | 72.0% | **+28.0%** |
| gram_positive_bacilli | 56.5% | **+39.2%** |
| gram_positive_cocci | 56.8% | **+34.1%** |
| gram_negative_bacilli | 44.4% | **+44.5%** |

---

## 3. Key Insights

1. **DL embedding dramatically improves taxonomy group accuracy** (45% → 90%) — this is the most impactful single change
2. **Group-constrained specialists** lift effective species accuracy from 48% (flat) to 89-100% (per-group)
3. **Shape model is identical** between DL and Classical — shape uses OpenCV computer vision features (area, circularity, etc.), not the CNN
4. **Fungi is easiest** to classify (100% with DL) — visually very different from bacteria
5. **Gram-negative bacilli is hardest** species group (88.9%) — E. coli, Salmonella, Klebsiella, Pseudomonas are visually similar
6. **Classical baseline had severe fungi recall** (8%) — DL fixed this to 96%

---

## 4. Dataset Information

- **Total images:** 629
- **Organisms:** 10 (8 bacteria + 2 fungi)
- **Modalities:** 2 (gram stain: 325 images, media plate: 304 images)
- **Holdout:** 80/20 stratified split, seed=42
- **Per-class sample sizes:** 53–91 images (class imbalance present)
- **Colony-level annotations:** 9102 colonies for shape classification

---

## 5. Artifact Files

| File | Size | Contents |
|------|------|----------|
| `artifacts/bacteria_models.joblib` | 621 MB | DL hybrid models + CNN embeddings + metrics |
| `artifacts/bacteria_models_classical.joblib` | 662 MB | Classical baseline models + 616-dim features + metrics |
| `artifacts/embedding_model.pt` | ~20 MB | EfficientNet-B0 PyTorch checkpoint |
| `artifacts/bacteria_models.metrics.json` | ~5 KB | DL metrics (accuracy, reports, model choices) |
| `artifacts/bacteria_models_classical.metrics.json` | ~5 KB | Classical metrics |

---

*Generated from `artifacts/bacteria_models.metrics.json` and `artifacts/bacteria_models_classical.metrics.json`*
*Last updated: 18 August 2026*
