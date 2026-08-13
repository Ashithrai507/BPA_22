from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch

INPUT_SIZE = 224

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

Transform = Callable[[np.ndarray], torch.Tensor]


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - _MEAN) / _STD


def _center_crop(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == size and w == size:
        return image
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return image[top : top + size, left : left + size]


def inference_transform(input_size: int = INPUT_SIZE) -> Transform:
    def apply(image: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
        cropped = _center_crop(resized, input_size)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return _normalize(tensor.contiguous())

    return apply


def train_transform(seed: int | None = None, input_size: int = INPUT_SIZE) -> Transform:
    rng = np.random.default_rng(seed)

    def apply(image: np.ndarray) -> torch.Tensor:
        h, w = image.shape[:2]
        scale = rng.uniform(0.8, 1.0)
        target = max(input_size, round(min(h, w) * scale))
        resized = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)
        cropped = _center_crop(resized, input_size)

        if rng.random() < 0.5:
            cropped = np.fliplr(cropped)
        if rng.random() < 0.5:
            cropped = np.flipud(cropped)
        angle = rng.uniform(-15, 15)
        if abs(angle) > 0.5:
            matrix = cv2.getRotationMatrix2D((input_size / 2, input_size / 2), angle, 1.0)
            cropped = cv2.warpAffine(cropped, matrix, (input_size, input_size))

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb * rng.uniform(0.85, 1.15) + rng.normal(0, 0.02, size=rgb.shape)
        rgb = np.clip(rgb, 0.0, 1.0)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return _normalize(tensor.contiguous())

    return apply
