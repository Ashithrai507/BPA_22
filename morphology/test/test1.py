import os
import sys
import cv2

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.extract_features import extract_features

img = cv2.imread("data/colonies/cocci/Enterococcus faecalis_media plates_4.png_colony_0.png")  # pick any one colony

features = extract_features(img)

print(features)