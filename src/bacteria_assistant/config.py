from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    """Resolve a configurable path from the environment, falling back to default."""
    return Path(os.environ.get(name, default)).expanduser()


ORGANISM_METADATA = {
    # Gram-positive cocci
    "Staphylococcus aureus": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "cocci",
        "taxonomy_group": "gram_positive_cocci",
    },
    "Streptococcus pyogenes": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "cocci",
        "taxonomy_group": "gram_positive_cocci",
    },
    "Enterococcus faecalis": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "cocci",
        "taxonomy_group": "gram_positive_cocci",
    },
    # Gram-positive bacilli
    "Bacillus subtilis": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "bacilli",
        "taxonomy_group": "gram_positive_bacilli",
    },
    "Clostridium sporogenes": {
        "organism_type": "bacteria",
        "gram_label": "gram_positive",
        "shape_label": "bacilli",
        "taxonomy_group": "gram_positive_bacilli",
    },
    # Gram-negative bacilli
    "Escherichia coli": {
        "organism_type": "bacteria",
        "gram_label": "gram_negative",
        "shape_label": "bacilli",
        "taxonomy_group": "gram_negative_bacilli",
    },
    "Klebsiella pneumoniae": {
        "organism_type": "bacteria",
        "gram_label": "gram_negative",
        "shape_label": "bacilli",
        "taxonomy_group": "gram_negative_bacilli",
    },
    "Pseudomonas aeruginosa": {
        "organism_type": "bacteria",
        "gram_label": "gram_negative",
        "shape_label": "bacilli",
        "taxonomy_group": "gram_negative_bacilli",
    },
    # Fungi (non-bacterial)
    "Candida albicans": {
        "organism_type": "fungi",
        "gram_label": "non_bacterial_fungi",
        "shape_label": "fungal",
        "taxonomy_group": "fungi",
    },
    "Aspergillus niger": {
        "organism_type": "fungi",
        "gram_label": "non_bacterial_fungi",
        "shape_label": "fungal",
        "taxonomy_group": "fungi",
    },
}

SUPPORTED_SHAPES = ("cocci", "bacilli", "spiral", "fungal")

UNKNOWN_LABEL = "unknown"

# Heuristic cleaning for colony shape training labels (issue #10).
# A colony row is dropped when its geometry strongly contradicts the parent
# image's shape_label. Thresholds are deliberately conservative; tune with
# `make evaluate` on the shape model.
SHAPE_CLEANING_RULES = {
    "cocci": {"max_aspect_ratio": 2.0},
    "bacilli": {"min_aspect_ratio": 1.2},
    "fungal": {"max_solidity": 0.95},
}

# Rejection thresholds to flag out-of-distribution images.
MIN_ORGANISM_CONFIDENCE = 0.55
MIN_COLONY_CONFIDENCE = 0.45
MIN_COLONIES_FOR_VALID = 1
MIN_ORGANISM_TYPE_PROB = 0.6
MIN_EDGE_DENSITY = 0.015
MIN_LAPLACIAN_VAR = 25.0

ORGANISMS_BY_GROUP = {
    "gram_positive_cocci": [
        "Staphylococcus aureus",
        "Streptococcus pyogenes",
        "Enterococcus faecalis",
    ],
    "gram_positive_bacilli": [
        "Bacillus subtilis",
        "Clostridium sporogenes",
    ],
    "gram_negative_bacilli": [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
    ],
    "fungi": [
        "Candida albicans",
        "Aspergillus niger",
    ],
}

DATASET_ROOT = _env_path("BACTERIA_DATASET_ROOT", "data/dataset")
ARTIFACT_DIR = _env_path("BACTERIA_ARTIFACT_DIR", "artifacts")
MODEL_PATH = _env_path("BACTERIA_MODEL_PATH", str(ARTIFACT_DIR / "bacteria_models.joblib"))


# Bump whenever feature extraction changes; forces model artifact refresh.
FEATURE_VERSION = 2

# CLAHE illumination normalization (issue #8 - imaging modality confounder).
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = 8

# Photometric augmentation (training fold only).
AUGMENT_PER_IMAGE = 3
AUGMENT_BRIGHTNESS_SIGMA = 12.0
AUGMENT_CONTRAST_ALPHA = (0.85, 1.15)
AUGMENT_CLAHE_CLIP_RANGE = (1.0, 3.0)


def normalize_organism_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").split()).strip()
