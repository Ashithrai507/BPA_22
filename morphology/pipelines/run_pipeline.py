import os
import sys

import cv2

# FIX IMPORT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MAX_AREA, MIN_AREA
from src.extraction.extract import extract_colonies
from src.preprocessing.preprocess import preprocess_image
from src.segmentation.segment import segment_image

INPUT_DIR = "data/raw/petri_dish/"
COLONY_BASE_DIR = "data/colonies/"
DEBUG_DIR = "output/debug/"

os.makedirs(DEBUG_DIR, exist_ok=True)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

print("\n🚀 Starting Pipeline...\n")

total_images = 0
total_colonies = 0

for root, _dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        path = os.path.join(root, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ Skipped: {filename}")
            continue

        # Label = folder name
        label = os.path.basename(root).lower()

        if label == "petri_dish":
            label = "unknown"

        # -------------------------
        # PROCESSING
        # -------------------------
        gray, blur = preprocess_image(img)
        mask = segment_image(blur)
        contours, colonies = extract_colonies(img, mask, MIN_AREA, MAX_AREA)

        # -------------------------
        # DEBUG IMAGE
        # -------------------------
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

        debug_name = f"{label}_{filename}"
        cv2.imwrite(os.path.join(DEBUG_DIR, debug_name), debug_img)

        # -------------------------
        # SAVE COLONIES BY CLASS
        # -------------------------
        label_dir = os.path.join(COLONY_BASE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        for i, colony in enumerate(colonies):
            colony_name = f"{filename}_colony_{i}.png"
            save_path = os.path.join(label_dir, colony_name)

            cv2.imwrite(save_path, colony)

        print(f"✅ {filename} ({label}) → {len(colonies)} colonies")

        total_images += 1
        total_colonies += len(colonies)

# -------------------------
# SUMMARY
# -------------------------
print("\n📊 Pipeline Summary:")
print(f"Total Images: {total_images}")
print(f"Total Colonies: {total_colonies}")
print("\n✅ Done!\n")
