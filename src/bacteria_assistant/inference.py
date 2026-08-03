from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    MIN_COLONIES_FOR_VALID,
    MIN_COLONY_CONFIDENCE,
    MIN_EDGE_DENSITY,
    MIN_LAPLACIAN_VAR,
    MIN_ORGANISM_CONFIDENCE,
    MIN_ORGANISM_TYPE_PROB,
    MODEL_PATH,
    ORGANISM_METADATA,
    ORGANISMS_BY_GROUP,
    UNKNOWN_LABEL,
)
from .features import (
    colony_measurement_to_json,
    colony_to_feature_dict,
    detect_distribution,
    extract_colonies,
    extract_image_features,
    read_image,
)


def load_models(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    return joblib.load(model_path)


def _shape_by_heuristic(shape_from_model: str, aspect_ratio: float, circularity: float) -> str:
    # Spiral is inferred by morphology heuristics because no spiral labels exist in current dataset.
    if aspect_ratio >= 3.0 and circularity <= 0.35:
        return "spiral"
    return shape_from_model


def _dominant_shape_and_confidence(shape_predictions: list[str], probs: list[float]) -> tuple[str, float]:
    if not shape_predictions:
        return "fungal", 0.0

    counts = Counter(shape_predictions)
    dominant_shape, count = counts.most_common(1)[0]
    dominance_ratio = count / max(1, len(shape_predictions))
    mean_prob = float(np.mean(probs)) if probs else 0.0

    confidence = float((0.65 * dominance_ratio) + (0.35 * mean_prob))
    confidence = max(0.0, min(1.0, confidence))
    return dominant_shape, confidence


def _should_reject_image(
    organism_type_prob: float,
    organism_confidence: float,
    shape_confidence: float,
    colony_count: int,
    edge_density: float,
    laplacian_var: float,
) -> bool:
    if colony_count < MIN_COLONIES_FOR_VALID:
        return True

    low_model_confidence = (organism_confidence < MIN_ORGANISM_CONFIDENCE) and (
        shape_confidence < MIN_COLONY_CONFIDENCE
    )
    low_organism_type = organism_type_prob < MIN_ORGANISM_TYPE_PROB
    low_image_quality = (edge_density < MIN_EDGE_DENSITY) and (laplacian_var < MIN_LAPLACIAN_VAR)

    return low_organism_type or low_model_confidence or low_image_quality


def _predict_species_with_group_constraint(
    artifacts: dict[str, Any],
    organism_model: Any,
    image_vector: pd.DataFrame,
    predicted_group: str,
) -> tuple[str, float]:
    group_species_models = artifacts.get("group_species_models", {})
    specialized = group_species_models.get(predicted_group)
    if specialized is not None:
        specialized_pred = str(specialized.predict(image_vector)[0])
        if hasattr(specialized, "predict_proba"):
            specialized_prob = float(np.max(specialized.predict_proba(image_vector)[0]))
        else:
            specialized_prob = 0.5
        return specialized_pred, specialized_prob

    fallback_pred = str(organism_model.predict(image_vector)[0])
    if not hasattr(organism_model, "predict_proba"):
        return fallback_pred, 0.5

    classes = [str(c) for c in organism_model.classes_]
    probs = organism_model.predict_proba(image_vector)[0]
    prob_map = {cls: float(prob) for cls, prob in zip(classes, probs, strict=False)}

    candidates = [org for org in ORGANISMS_BY_GROUP.get(predicted_group, []) if org in prob_map]
    if not candidates:
        return fallback_pred, float(np.max(probs))

    best = max(candidates, key=lambda org: prob_map[org])
    return best, prob_map[best]


def predict_bacteria_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    mode: str = "basic",
) -> dict[str, Any]:
    model_path = Path(model_path) if model_path else MODEL_PATH
    artifacts = load_models(model_path)

    image = read_image(str(image_path))

    image_features = extract_image_features(image)
    image_vector = pd.DataFrame([image_features])[artifacts["image_feature_columns"]]

    organism_metadata = artifacts.get("organism_metadata", ORGANISM_METADATA)

    organism_type_model = artifacts.get("organism_type_model")
    if organism_type_model is None:
        raise ValueError(
            "Loaded artifact does not contain `organism_type_model`. "
            "Please retrain the model by running train_model.py."
        )

    predicted_organism_type = str(organism_type_model.predict(image_vector)[0])
    organism_type_prob = 0.5
    if hasattr(organism_type_model, "predict_proba"):
        organism_type_prob = float(np.max(organism_type_model.predict_proba(image_vector)[0]))

    group_model = artifacts.get("group_model")
    if group_model is None:
        raise ValueError(
            "Loaded artifact does not contain `group_model`. " "Please retrain the model by running train_model.py."
        )

    predicted_group = str(group_model.predict(image_vector)[0])

    organism_model = artifacts.get("organism_model")
    if organism_model is None:
        raise ValueError(
            "Loaded artifact does not contain `organism_model`. Please retrain the model by running train_model.py."
        )

    predicted_bacteria_name, organism_confidence = _predict_species_with_group_constraint(
        artifacts,
        organism_model,
        image_vector,
        predicted_group,
    )

    gram_model = artifacts["gram_model"]
    gram_label = str(gram_model.predict(image_vector)[0])

    meta = organism_metadata.get(predicted_bacteria_name, {})
    if meta:
        predicted_organism_type = str(meta.get("organism_type", predicted_organism_type))
        gram_label = str(meta.get("gram_label", gram_label))

    colonies = extract_colonies(image)
    colony_json_rows: list[dict[str, Any]] = []
    shape_preds: list[str] = []
    shape_probs: list[float] = []

    shape_model = artifacts["shape_model"]
    feature_cols = artifacts["colony_feature_columns"]

    for idx, colony in enumerate(colonies, start=1):
        feature_row = colony_to_feature_dict(colony)
        row_df = pd.DataFrame([feature_row])[feature_cols]
        pred = str(shape_model.predict(row_df)[0])

        model_prob = 0.5
        if hasattr(shape_model, "predict_proba"):
            prob_vec = shape_model.predict_proba(row_df)[0]
            model_prob = float(np.max(prob_vec))

        pred = _shape_by_heuristic(pred, colony.aspect_ratio, colony.circularity)

        shape_preds.append(pred)
        shape_probs.append(model_prob)

        colony_json_rows.append(colony_measurement_to_json(colony, idx, pred))

    dominant_shape, confidence = _dominant_shape_and_confidence(shape_preds, shape_probs)
    if predicted_organism_type == "fungi":
        dominant_shape = "fungal"
    final_confidence = max(confidence, organism_confidence)

    if _should_reject_image(
        organism_type_prob,
        organism_confidence,
        confidence,
        len(colony_json_rows),
        float(image_features.get("edge_density", 0.0)),
        float(image_features.get("laplacian_var", 0.0)),
    ):
        unknown_response = {
            "organism_type": UNKNOWN_LABEL,
            "predicted_bacteria_name": UNKNOWN_LABEL,
            "bacteria_type": UNKNOWN_LABEL,
            "total_colonies_detected": len(colony_json_rows),
            "dominant_shape": UNKNOWN_LABEL,
            "confidence": round(final_confidence, 2),
        }
        if mode == "basic":
            return unknown_response

        return {
            "organism_type": UNKNOWN_LABEL,
            "predicted_bacteria_name": UNKNOWN_LABEL,
            "bacteria_type": UNKNOWN_LABEL,
            "total_colonies": len(colony_json_rows),
            "colonies": colony_json_rows,
            "final_morphology": {
                "dominant_shape": UNKNOWN_LABEL,
                "distribution": UNKNOWN_LABEL,
                "confidence": round(final_confidence, 2),
            },
        }

    if mode == "basic":
        return {
            "organism_type": predicted_organism_type,
            "predicted_bacteria_name": predicted_bacteria_name,
            "bacteria_type": gram_label,
            "total_colonies_detected": len(colony_json_rows),
            "dominant_shape": dominant_shape,
            "confidence": round(final_confidence, 2),
        }

    if mode != "advanced":
        raise ValueError("mode must be either 'basic' or 'advanced'")

    return {
        "organism_type": predicted_organism_type,
        "predicted_bacteria_name": predicted_bacteria_name,
        "bacteria_type": gram_label,
        "total_colonies": len(colony_json_rows),
        "colonies": colony_json_rows,
        "final_morphology": {
            "dominant_shape": dominant_shape,
            "distribution": detect_distribution(colonies, image.shape),
            "confidence": round(final_confidence, 2),
        },
    }
