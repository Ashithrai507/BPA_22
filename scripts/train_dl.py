from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import DATASET_ROOT
from bacteria_assistant.dl.dataset import build_image_dataset
from bacteria_assistant.dl.model import EmbeddingModel
from bacteria_assistant.dl.trainer import (
    TrainingConfig,
    fit_embedding_model,
    get_device,
)
from bacteria_assistant.training import _load_labeled_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DL embedding model.")
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
        help="Workspace root path (for resolving relative image paths)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "embedding_model.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frozen-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda")
    return parser.parse_args()


def _resolve_image_path(workspace_root: Path, image_path: str) -> str:
    candidate = workspace_root / image_path
    if candidate.exists():
        return str(candidate)
    if Path(image_path).exists():
        return str(image_path)
    raise FileNotFoundError(f"Image path does not exist: {image_path}")


def main() -> None:
    args = parse_args()
    labeled_df = _load_labeled_dataframe(args.dataset_csv)
    labeled_df.loc[:, "image_path"] = labeled_df["image_path"].map(
        lambda p: _resolve_image_path(args.workspace_root, str(p))
    )

    dataset = build_image_dataset(labeled_df)
    model = EmbeddingModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_species=len(dataset.encoders["species"]),
        num_groups=len(dataset.encoders["group"]),
        num_grams=len(dataset.encoders["gram"]),
        num_types=len(dataset.encoders["type"]),
    )

    device = args.device or get_device()
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        frozen_epochs=args.frozen_epochs,
        input_size=args.input_size,
        aux_weight=args.aux_weight,
        contrastive_weight=args.contrastive_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, metadata = fit_embedding_model(
        model=model,
        dataset=dataset,
        config=config,
        checkpoint_path=args.output,
        device=device,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
