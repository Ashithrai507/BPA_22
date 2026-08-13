from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from bacteria_assistant.config import DATASET_ROOT, MODEL_PATH, ORGANISMS_BY_GROUP
from bacteria_assistant.inference import load_models
from bacteria_assistant.training import _load_labeled_dataframe, _resolve_image_path

TARGETS = {
    "species": 0.55,
    "group": 0.60,
    "organism_type": 0.85,
    "gram": 0.72,
}


def _species_prediction(probs: dict[str, float], predicted_group: str) -> str:
    candidates = [org for org in ORGANISMS_BY_GROUP.get(predicted_group, []) if org in probs]
    if candidates:
        return max(candidates, key=lambda org: probs[org])
    return max(probs, key=probs.get)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate inference-time DL accuracy on the shared 80/20 holdout (seed 42)."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / MODEL_PATH,
        help="Path to the trained DL joblib bundle",
    )
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
    args = parser.parse_args()

    from bacteria_assistant.dl.extract import extract_embedding, species_probabilities

    labeled_df = _load_labeled_dataframe(args.dataset_csv)
    _, test_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=labeled_df["organism"],
    )

    artifacts = load_models(args.artifact)
    embedding_model = artifacts.get("_embedding_model")
    if embedding_model is None:
        print(f"error: artifact has no DL embedding model attached: {args.artifact}", file=sys.stderr)
        return 1

    emb_cols = artifacts["image_feature_columns"]
    species_encoder = artifacts["_embedding_encoders"]["species"]
    input_size = artifacts["_embedding_input_size"]
    group_model = artifacts["group_model"]
    type_model = artifacts["organism_type_model"]
    gram_model = artifacts["gram_model"]

    counts = {"species": 0, "group": 0, "organism_type": 0, "gram": 0}
    n = 0
    for _, sample in test_df.iterrows():
        image_path = _resolve_image_path(args.workspace_root, str(sample["image_path"]))
        emb = extract_embedding(embedding_model, image_path, input_size=input_size)
        vec = pd.DataFrame([dict(zip(emb_cols, emb.tolist(), strict=False))])[emb_cols]

        predicted_group = str(group_model.predict(vec)[0])
        predicted_type = str(type_model.predict(vec)[0])
        predicted_gram = str(gram_model.predict(vec)[0])
        probs = species_probabilities(
            embedding_model,
            image_path,
            species_encoder,
            input_size=input_size,
            tta=True,
        )
        predicted_species = _species_prediction(probs, predicted_group)

        counts["species"] += int(predicted_species == sample["organism"])
        counts["group"] += int(predicted_group == sample["taxonomy_group"])
        counts["organism_type"] += int(predicted_type == sample["organism_type"])
        counts["gram"] += int(predicted_gram == sample["gram_label"])
        n += 1

    metrics = {name: round(count / n, 4) for name, count in counts.items()}

    shape_accuracy = None
    metrics_path = args.artifact.with_suffix(".metrics.json")
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as handle:
            shape_accuracy = round(json.load(handle)["shape_metrics"]["accuracy"], 4)

    print(f"Holdout evaluation (n={n} images, DL species head + TTA + group constraint)")
    print(f"{'Modality':<14} {'Accuracy':<10} {'Target':<10} Status")
    print("-" * 50)
    ok = True
    for name, target in TARGETS.items():
        acc = metrics[name]
        passed = acc >= target
        ok &= passed
        print(f"{name:<14} {acc:<10.4f} {target:<10.2f} {'PASS' if passed else 'FAIL'}")
    if shape_accuracy is not None:
        print(f"{'shape':<14} {shape_accuracy:<10.4f} {'n/a':<10} (from training metrics)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
