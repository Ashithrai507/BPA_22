from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protein_engine import pipeline
from protein_engine.config import DEFAULT_TOP_N, PATHS
from protein_engine.retrieval.api_cache import ApiError, CacheMissError
from protein_engine.taxonomy.resolver import ResolutionError


def _predict_species(image_path: Path) -> str:
    try:
        from bacteria_assistant.config import MODEL_PATH
        from bacteria_assistant.inference import predict_bacteria_image
    except ImportError as exc:
        raise ResolutionError("CV pipeline unavailable; use --species instead") from exc
    result = predict_bacteria_image(
        image_path=image_path,
        model_path=PROJECT_ROOT / MODEL_PATH,
        mode="basic",
    )
    name = result.get("predicted_bacteria_name")
    if not name or name == "unknown":
        raise ResolutionError(f"could not predict a species from image: {image_path}")
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank folding-ready proteins for a species.")
    parser.add_argument("--species", type=str, help="Scientific species name")
    parser.add_argument("--image", type=Path, help="Petri dish image (predicts species via CV)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of proteins to return")
    parser.add_argument("--output-dir", type=Path, default=PATHS.output_dir)
    parser.add_argument("--cache-dir", type=Path, default=PATHS.cache_dir)
    parser.add_argument("--reference-dir", type=Path, default=PATHS.reference_dir)
    parser.add_argument("--offline", action="store_true", help="Serve from cache only")
    args = parser.parse_args(argv)

    if not args.species and args.image is None:
        parser.error("either --species or --image is required")

    try:
        species = args.species or _predict_species(args.image)
        payload = pipeline.run(
            species,
            top_n=args.top_n,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            reference_dir=args.reference_dir,
            offline=args.offline,
        )
    except ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ApiError, CacheMissError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
