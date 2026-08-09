from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import DATASET_ROOT, MODEL_PATH
from bacteria_assistant.inference import predict_bacteria_image


def _model_path() -> Path:
    path = PROJECT_ROOT / MODEL_PATH
    if not path.exists():
        pytest.skip(f"Trained model artifact not found at {path}. Run `make train` first.")
    return path


def _first_image() -> Path:
    dataset_roots = [PROJECT_ROOT / DATASET_ROOT, PROJECT_ROOT / "Bacteria dataset"]
    for dataset_root in dataset_roots:
        if dataset_root.exists():
            for path in dataset_root.rglob("*.png"):
                return path
    pytest.skip(f"Dataset root not found in {dataset_roots}. See CONTRIBUTING.md.")


def test_basic_output_contract() -> None:
    model_path = _model_path()

    output = predict_bacteria_image(_first_image(), model_path=model_path, mode="basic")

    required_keys = {
        "organism_type",
        "predicted_bacteria_name",
        "bacteria_type",
        "total_colonies_detected",
        "dominant_shape",
        "confidence",
    }
    assert required_keys.issubset(set(output.keys()))
    assert output["organism_type"] in {"bacteria", "fungi", "unknown"}
    assert isinstance(output["predicted_bacteria_name"], str)
    assert output["predicted_bacteria_name"]
    assert output["bacteria_type"] in {"gram_positive", "gram_negative", "non_bacterial_fungi", "unknown"}
    assert isinstance(output["total_colonies_detected"], int)
    assert output["dominant_shape"] in {"cocci", "bacilli", "spiral", "fungal", "unknown"}
    assert 0.0 <= output["confidence"] <= 1.0


def test_advanced_output_contract() -> None:
    model_path = _model_path()

    output = predict_bacteria_image(_first_image(), model_path=model_path, mode="advanced")

    required_keys = {
        "organism_type",
        "predicted_bacteria_name",
        "bacteria_type",
        "total_colonies",
        "colonies",
        "final_morphology",
    }
    assert required_keys.issubset(set(output.keys()))
    assert output["organism_type"] in {"bacteria", "fungi", "unknown"}
    assert isinstance(output["predicted_bacteria_name"], str)
    assert output["predicted_bacteria_name"]
    assert output["bacteria_type"] in {"gram_positive", "gram_negative", "non_bacterial_fungi", "unknown"}
    assert isinstance(output["total_colonies"], int)

    final_morphology = output["final_morphology"]
    assert set(final_morphology.keys()) == {"dominant_shape", "distribution", "confidence"}
    assert final_morphology["dominant_shape"] in {"cocci", "bacilli", "spiral", "fungal", "unknown"}
    assert isinstance(final_morphology["distribution"], str)
    assert 0.0 <= final_morphology["confidence"] <= 1.0
