from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import (
    AUGMENT_BRIGHTNESS_SIGMA,
    AUGMENT_CLAHE_CLIP_RANGE,
    AUGMENT_CONTRAST_ALPHA,
    AUGMENT_PER_IMAGE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID))
    return clahe.apply(gray)


def augment_image(image: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    variants: list[np.ndarray] = []
    for _ in range(AUGMENT_PER_IMAGE):
        variant = image.copy().astype(np.float32)

        alpha = rng.uniform(*AUGMENT_CONTRAST_ALPHA)
        beta = rng.normal(0.0, AUGMENT_BRIGHTNESS_SIGMA)
        variant = np.clip(variant * alpha + beta, 0, 255).astype(np.uint8)

        clip = rng.uniform(*AUGMENT_CLAHE_CLIP_RANGE)
        lab = cv2.cvtColor(variant, cv2.COLOR_BGR2LAB)
        tile = (CLAHE_TILE_GRID, CLAHE_TILE_GRID)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(lab[:, :, 0])
        variant = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        variants.append(variant)
    return variants


@dataclass
class ColonyMeasurement:
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    solidity: float
    equivalent_diameter: float
    mean_intensity: float
    centroid_x: float
    centroid_y: float


def read_image(image_path: str) -> np.ndarray:
    """Read image robustly for paths containing spaces/unicode."""
    raw = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def extract_image_features(image: np.ndarray) -> dict[str, float]:
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = _apply_clahe(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    features: dict[str, float] = {}

    for idx, channel_name in enumerate(("r", "g", "b")):
        ch = rgb[:, :, idx]
        features[f"{channel_name}_mean"] = float(np.mean(ch))
        features[f"{channel_name}_std"] = float(np.std(ch))

    for idx, channel_name in enumerate(("h", "s", "v")):
        ch = hsv[:, :, idx]
        features[f"{channel_name}_mean"] = float(np.mean(ch))
        features[f"{channel_name}_std"] = float(np.std(ch))

    features["gray_mean"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features["laplacian_var"] = float(lap_var)

    edges = cv2.Canny(gray, 70, 160)
    features["edge_density"] = float(np.mean(edges > 0))

    for idx, channel_name in enumerate(("r", "g", "b")):
        hist = cv2.calcHist([rgb], [idx], None, [8], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        for i, value in enumerate(hist):
            features[f"{channel_name}_hist_{i}"] = float(value)

    # Add compact spatial signature to improve species-level discrimination.
    spatial = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    for i, value in enumerate(spatial.flatten()):
        features[f"gray_spatial_{i}"] = float(value)

    features.update(_colony_shape_features(gray))
    features.update(_gradient_texture_features(gray))

    return features


def _hu_log_scale(hu: np.ndarray) -> np.ndarray:
    return np.sign(hu) * np.log1p(np.abs(hu))


def _colony_shape_features(gray: np.ndarray) -> dict[str, float]:
    contours = _choose_best_mask(gray)
    features: dict[str, float] = {"colony_count": float(len(contours))}
    for i in range(7):
        features[f"hu_{i}"] = 0.0
        features[f"colony_hu_mean_{i}"] = 0.0
        features[f"colony_hu_std_{i}"] = 0.0
    if not contours:
        return features

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    image_hu = _hu_log_scale(cv2.HuMoments(cv2.moments(mask)).flatten())
    for i, value in enumerate(image_hu):
        features[f"hu_{i}"] = float(value)

    per_colony = np.stack([_hu_log_scale(cv2.HuMoments(cv2.moments(c)).flatten()) for c in contours])
    mean = per_colony.mean(axis=0)
    std = per_colony.std(axis=0)
    for i in range(7):
        features[f"colony_hu_mean_{i}"] = float(mean[i])
        features[f"colony_hu_std_{i}"] = float(std[i])
    return features


def _gradient_texture_features(gray: np.ndarray) -> dict[str, float]:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    r = np.hypot(np.mean(np.cos(angle)), np.mean(np.sin(angle)))
    angle_std = float(np.sqrt(max(-2.0 * np.log(r + 1e-12), 0.0)))
    return {
        "grad_mag_mean": float(np.mean(mag)),
        "grad_mag_std": float(np.std(mag)),
        "grad_angle_std": angle_std,
    }


def _threshold_mask(gray: np.ndarray, invert: bool) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _valid_contours(mask: np.ndarray, image_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid: list[np.ndarray] = []
    min_area = max(20.0, image_area * 0.00003)
    max_area = image_area * 0.12
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid.append(contour)
    return valid


def _choose_best_mask(gray: np.ndarray) -> list[np.ndarray]:
    image_area = float(gray.shape[0] * gray.shape[1])
    contours_inv = _valid_contours(_threshold_mask(gray, invert=True), image_area)
    contours_plain = _valid_contours(_threshold_mask(gray, invert=False), image_area)

    # Pick segmentation with realistic but richer contour set.
    if len(contours_inv) == 0 and len(contours_plain) == 0:
        return []
    if len(contours_inv) == 0:
        return contours_plain
    if len(contours_plain) == 0:
        return contours_inv
    return contours_inv if len(contours_inv) >= len(contours_plain) else contours_plain


def _safe_mean_intensity(gray: np.ndarray, contour: np.ndarray) -> float:
    contour_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    pixel_values = gray[contour_mask == 255]
    if pixel_values.size == 0:
        return float(np.mean(gray))
    return float(np.mean(pixel_values))


def _contour_to_measurement(contour: np.ndarray, gray: np.ndarray) -> ColonyMeasurement:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float((4.0 * np.pi * area) / (perimeter**2 + 1e-8))

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w / (h + 1e-8))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / (hull_area + 1e-8))

    equivalent_diameter = float(np.sqrt((4.0 * area) / (np.pi + 1e-8)))
    mean_intensity = _safe_mean_intensity(gray, contour)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        cx = float(x + (w / 2.0))
        cy = float(y + (h / 2.0))
    else:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

    return ColonyMeasurement(
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
        equivalent_diameter=equivalent_diameter,
        mean_intensity=mean_intensity,
        centroid_x=cx,
        centroid_y=cy,
    )


def extract_colonies(image: np.ndarray) -> list[ColonyMeasurement]:
    gray = _apply_clahe(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    contours = _choose_best_mask(gray)
    colonies = [_contour_to_measurement(contour, gray) for contour in contours]
    colonies.sort(key=lambda c: c.area, reverse=True)
    return colonies


def colony_to_feature_dict(colony: ColonyMeasurement) -> dict[str, float]:
    return {
        "area": colony.area,
        "perimeter": colony.perimeter,
        "circularity": colony.circularity,
        "aspect_ratio": colony.aspect_ratio,
        "solidity": colony.solidity,
        "equivalent_diameter": colony.equivalent_diameter,
        "mean_intensity": colony.mean_intensity,
    }


def detect_distribution(colonies: list[ColonyMeasurement], image_shape: tuple[int, int, int]) -> str:
    if len(colonies) < 2:
        return "isolated"

    h, w = image_shape[:2]
    diag = float(np.sqrt((h * h) + (w * w)))
    points = np.array([[c.centroid_x, c.centroid_y] for c in colonies], dtype=np.float64)

    dists: list[float] = []
    for i in range(len(points)):
        delta = points - points[i]
        pairwise = np.sqrt(np.sum(delta * delta, axis=1))
        pairwise = pairwise[pairwise > 0]
        if pairwise.size > 0:
            dists.append(float(np.min(pairwise)))

    if not dists:
        return "isolated"

    normalized = float(np.mean(dists) / (diag + 1e-8))
    if normalized < 0.08:
        return "clustered"
    if normalized < 0.16:
        return "mixed"
    return "dispersed"


def colony_measurement_to_json(colony: ColonyMeasurement, colony_id: int, shape: str) -> dict[str, Any]:
    return {
        "id": colony_id,
        "area": round(colony.area, 2),
        "perimeter": round(colony.perimeter, 2),
        "circularity": round(colony.circularity, 2),
        "aspect_ratio": round(colony.aspect_ratio, 2),
        "solidity": round(colony.solidity, 2),
        "equivalent_diameter": round(colony.equivalent_diameter, 2),
        "mean_intensity": round(colony.mean_intensity, 2),
        "predicted_shape": shape,
    }
