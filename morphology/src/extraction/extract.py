import math

import cv2

from src.config import MAX_ASPECT_RATIO, MIN_ASPECT_RATIO, MIN_CIRCULARITY, MIN_HEIGHT, MIN_WIDTH


def extract_colonies(img, mask, min_area, max_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    colonies = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Area filter
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Size filter
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        # Aspect ratio filter
        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # Circularity filter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        # Padding (safe)
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)

        colony = img[y : y + h, x : x + w]

        filtered_contours.append(cnt)
        colonies.append(colony)

    return filtered_contours, colonies
