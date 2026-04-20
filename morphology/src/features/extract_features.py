import cv2
import numpy as np
import math

def extract_features(colony_img):

    gray = cv2.cvtColor(colony_img, cv2.COLOR_BGR2GRAY)

    # Binary mask for shape
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # -------------------------
    # SHAPE FEATURES
    # -------------------------
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return None

    circularity = 4 * math.pi * area / (perimeter * perimeter)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area != 0 else 0

    # -------------------------
    # SIZE FEATURES
    # -------------------------
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # -------------------------
    # TEXTURE FEATURES
    # -------------------------
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    pixels = gray[mask == 255]

    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)

    # -------------------------
    # FINAL FEATURE VECTOR
    # -------------------------
    features = {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "equivalent_diameter": equivalent_diameter,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity
    }

    return features