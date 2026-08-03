import cv2
import numpy as np


def segment_image(blur):
    # Otsu threshold
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Choose better threshold automatically
    if otsu.mean() < 127:
        thresh = otsu
    else:
        thresh = adaptive

    # Ensure colonies are white
    if thresh.mean() > 127:
        thresh = cv2.bitwise_not(thresh)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

    return closing
