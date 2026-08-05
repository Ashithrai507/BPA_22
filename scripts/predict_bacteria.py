from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import MODEL_PATH
from bacteria_assistant.inference import predict_bacteria_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bacteria morphology from an image.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / MODEL_PATH,
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="advanced",
        choices=["basic", "advanced"],
        help="Output format mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_bacteria_image(image_path=args.image, model_path=args.model, mode=args.mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
