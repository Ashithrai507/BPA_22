from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .dataset import LabelEncoder
from .model import EmbeddingModel
from .transforms import inference_transform


def load_embedding_model(
    checkpoint_path: Path,
) -> tuple[EmbeddingModel, dict[str, LabelEncoder], dict[str, Any]]:
    """Load an embedding model, its label encoders, and config from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = EmbeddingModel(
        backbone_name=config.get("backbone", "efficientnet_b0"),
        pretrained=False,
        num_species=len(ckpt["label_encoders"]["species"]["labels"]),
        num_groups=len(ckpt["label_encoders"]["group"]["labels"]),
        num_grams=len(ckpt["label_encoders"]["gram"]["labels"]),
        num_types=len(ckpt["label_encoders"]["type"]["labels"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    encoders = {key: LabelEncoder(meta["labels"]) for key, meta in ckpt["label_encoders"].items()}
    return model, encoders, config


def extract_embedding(
    model: EmbeddingModel,
    image_path: Path,
    device: str = "cpu",
    input_size: int = 224,
) -> torch.Tensor:
    """Return the embedding for a single image."""
    image = _read_image(image_path)
    transform = inference_transform(input_size)
    tensor = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    return out["embeddings"].squeeze(0).cpu()


def predict_species(
    model: EmbeddingModel,
    image_path: Path,
    species_encoder: LabelEncoder,
    device: str = "cpu",
    input_size: int = 224,
    tta: bool = False,
) -> tuple[str, float]:
    """Predict species via the DL species head. Returns (species_name, confidence)."""
    probs = species_probabilities(
        model,
        image_path,
        species_encoder,
        device=device,
        input_size=input_size,
        tta=tta,
    )
    name = max(probs, key=probs.get)
    return name, probs[name]


def species_probabilities(
    model: EmbeddingModel,
    image_path: Path,
    species_encoder: LabelEncoder,
    device: str = "cpu",
    input_size: int = 224,
    tta: bool = False,
) -> dict[str, float]:
    """Return softmax species probabilities keyed by species name (TTA-aware).

    When tta=True, average softmax probabilities over the original image, a
    horizontal flip, and a small number of scale-jitter crops.
    """
    image = _read_image(image_path)
    model = model.to(device)
    model.eval()

    def forward(tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(tensor)
        return torch.softmax(out["logits"]["species"].squeeze(0), dim=0)

    base = inference_transform(input_size)
    prob_sum = forward(base(image).unsqueeze(0).to(device))
    n = 1

    if tta:
        flipped = np.fliplr(image)
        prob_sum += forward(base(flipped).unsqueeze(0).to(device))
        n += 1

        h, w = image.shape[:2]
        for crop_scale in (0.85, 0.7):
            size = max(input_size, round(min(h, w) * crop_scale))
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            crop = _center_crop(resized, input_size)
            prob_sum += forward(base(crop).unsqueeze(0).to(device))
            n += 1

    probs = prob_sum / n
    return {species_encoder.decode(i): float(probs[i].item()) for i in range(len(species_encoder))}


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return image[top : top + size, left : left + size]


def _read_image(image_path: Path):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image
