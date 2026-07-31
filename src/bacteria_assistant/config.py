from __future__ import annotations

from pathlib import Path

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

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "bacteria_models.joblib"

def normalize_organism_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").split()).strip()
