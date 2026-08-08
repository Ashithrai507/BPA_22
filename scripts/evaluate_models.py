from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import MODEL_PATH


def _per_class_rows(report: dict) -> list[tuple[str, float, float, float, int]]:
    rows: list[tuple[str, float, float, float, int]] = []
    for label, values in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            (
                str(label),
                float(values["precision"]),
                float(values["recall"]),
                float(values["f1-score"]),
                int(values["support"]),
            )
        )
    return rows


def _print_task(name: str, metrics: dict) -> None:
    print(f"\n## {name}")
    print(f"  accuracy         : {metrics.get('accuracy', 'n/a')}")
    print(f"  scoring method   : {metrics.get('scoring_method', 'n/a')}")
    if "balanced_accuracy" in metrics:
        print(f"  balanced accuracy: {metrics['balanced_accuracy']}")

    report = metrics.get("report")
    if report:
        print("  per-class precision/recall:")
        for label, precision, recall, f1, support in _per_class_rows(report):
            print(f"    {label:<16} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} n={support}")

    per_modality = metrics.get("per_modality_accuracy")
    if per_modality:
        print("  per-modality accuracy:")
        for modality, acc in per_modality.items():
            print(f"    {modality:<20} {acc:.3f}")


def _fungi_recall(metrics: dict) -> float | None:
    report = metrics.get("report") or {}
    fungi = report.get("fungi")
    if fungi is None:
        return None
    return float(fungi["recall"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Print model quality report from the metrics artifact.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / MODEL_PATH.with_suffix(".metrics.json"),
        help="Path to the training metrics JSON",
    )
    parser.add_argument(
        "--min-fungi-recall",
        type=float,
        default=None,
        help="Exit non-zero if organism_type fungi recall is below this threshold",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"error: metrics file not found: {args.metrics}", file=sys.stderr)
        return 1

    with args.metrics.open(encoding="utf-8") as handle:
        data = json.load(handle)

    print("Model quality report")
    print(f"  model path: {data.get('model_path', 'n/a')}")

    for key in (
        "organism_type_metrics",
        "gram_metrics",
        "group_metrics",
        "organism_metrics",
        "shape_metrics",
    ):
        if key in data:
            _print_task(key, data[key])

    for group_name, group_metrics in (data.get("group_species_metrics") or {}).items():
        _print_task(f"group_species_metrics [{group_name}]", group_metrics)

    organism_type_metrics = data.get("organism_type_metrics")
    if organism_type_metrics is not None:
        fungi_recall = _fungi_recall(organism_type_metrics)
        if fungi_recall is not None:
            print(f"\nFungi recall (organism_type): {fungi_recall:.3f}")

            if args.min_fungi_recall is not None:
                if fungi_recall < args.min_fungi_recall:
                    print(
                        f"error: fungi recall {fungi_recall:.3f} below threshold " f"{args.min_fungi_recall:.3f}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"  ok: fungi recall >= {args.min_fungi_recall}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
