from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import DATASET_ROOT, MODEL_PATH
from bacteria_assistant.training import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bacteria gram/shape prediction models.")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=PROJECT_ROOT / DATASET_ROOT / "dataset_full.csv",
        help="Path to dataset_full.csv",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Workspace root path",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / MODEL_PATH,
        help="Output path for trained model artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_models(
        dataset_csv=args.dataset_csv,
        workspace_root=args.workspace_root,
        model_output_path=args.output_model,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
