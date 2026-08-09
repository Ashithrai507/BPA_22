from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    AUGMENT_PER_IMAGE,
    FEATURE_VERSION,
    MODEL_PATH,
    ORGANISM_METADATA,
    SHAPE_CLEANING_RULES,
    SUPPORTED_SHAPES,
    normalize_organism_name,
)
from .features import (
    augment_image,
    colony_to_feature_dict,
    extract_colonies,
    extract_image_features,
    read_image,
)


def _resolve_image_path(workspace_root: Path, image_path: str) -> Path:
    candidate = workspace_root / image_path
    if candidate.exists():
        return candidate

    # CSVs may reference "Bacteria dataset/..." while images live under data/dataset/.
    relocated = Path(str(image_path).replace("Bacteria dataset/", "data/dataset/"))
    candidate = workspace_root / relocated
    if candidate.exists():
        return candidate

    # If CSV path is already rooted at workspace, fallback to direct path.
    path_obj = Path(image_path)
    if path_obj.exists():
        return path_obj

    raise FileNotFoundError(f"Image path does not exist: {image_path}")


def _load_labeled_dataframe(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv).copy(deep=True)
    df.loc[:, "organism"] = df["organism"].map(normalize_organism_name)
    df.loc[:, "imaging_type"] = df["imaging_type"].astype(str).str.strip().str.lower()

    labeled_df = df[df["organism"].isin(ORGANISM_METADATA.keys())].copy()
    if labeled_df.empty:
        raise ValueError("No rows found in dataset for supported organisms.")

    labeled_df.loc[:, "organism_type"] = labeled_df["organism"].map(
        lambda x: ORGANISM_METADATA[str(x)]["organism_type"]
    )
    labeled_df.loc[:, "gram_label"] = labeled_df["organism"].map(lambda x: ORGANISM_METADATA[str(x)]["gram_label"])
    labeled_df.loc[:, "shape_label"] = labeled_df["organism"].map(lambda x: ORGANISM_METADATA[str(x)]["shape_label"])
    labeled_df.loc[:, "taxonomy_group"] = labeled_df["organism"].map(
        lambda x: ORGANISM_METADATA[str(x)]["taxonomy_group"]
    )
    return labeled_df


def _balance_modalities_per_species(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Trim over-represented imaging modalities per species within the train fold (issue #9).

    Each species with 2+ modalities is downsized so both modalities contribute the
    minimum observed count (no replacement, seeded). Single-modality species are
    untouched. Returns the balanced DataFrame and per-species audit stats.
    """
    if train_df.empty or "imaging_type" not in train_df.columns:
        return train_df.copy(), {}

    rng = np.random.default_rng(random_state)
    balanced_parts: list[pd.DataFrame] = []
    stats: dict[str, dict[str, Any]] = {}

    for species, group in train_df.groupby("organism", sort=True):
        modalities = [str(m) for m in sorted(group["imaging_type"].unique())]
        before = {m: int((group["imaging_type"] == m).sum()) for m in modalities}
        if len(modalities) < 2:
            balanced_parts.append(group)
            stats[str(species)] = {"before": before, "after": dict(before), "balanced": False}
            continue

        target = min(before.values())
        kept: list[pd.DataFrame] = []
        for modality in modalities:
            sub = group[group["imaging_type"] == modality]
            if len(sub) > target:
                positions = rng.choice(np.arange(len(sub)), size=target, replace=False)
                sub = sub.iloc[np.sort(positions)]
            kept.append(sub)
        balanced = pd.concat(kept)
        balanced_parts.append(balanced)
        after = {m: int((balanced["imaging_type"] == m).sum()) for m in modalities}
        stats[str(species)] = {"before": before, "after": after, "balanced": True}

    result = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else train_df.copy()
    return result, stats


def _build_image_feature_table(
    labeled_df: pd.DataFrame,
    workspace_root: Path,
    augment: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in labeled_df.iterrows():
        image_path = _resolve_image_path(workspace_root, str(sample["image_path"]))
        image = read_image(str(image_path))

        meta = {
            "image_path": str(image_path),
            "organism": sample["organism"],
            "organism_type": sample["organism_type"],
            "gram_label": sample["gram_label"],
            "shape_label": sample["shape_label"],
            "taxonomy_group": sample["taxonomy_group"],
            "imaging_type": sample["imaging_type"],
        }

        if not augment:
            f = extract_image_features(image)
            f.update(meta)
            rows.append(f)
            continue

        # Deterministic per-image seed so runs are reproducible.
        seed = int(hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8], 16)
        variants = [image] + augment_image(image, seed)
        for idx, variant in enumerate(variants):
            f = extract_image_features(variant)
            f.update(meta)
            f["variant_id"] = idx
            rows.append(f)

    return pd.DataFrame(rows)


def _build_colony_feature_table(labeled_df: pd.DataFrame, workspace_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in labeled_df.iterrows():
        image_path = _resolve_image_path(workspace_root, str(sample["image_path"]))
        image = read_image(str(image_path))
        colonies = extract_colonies(image)

        if not colonies:
            continue

        image_shape_label = str(sample["shape_label"])
        for colony in colonies:
            features = colony_to_feature_dict(colony)
            features["shape_label"] = image_shape_label
            features["image_id"] = str(image_path)
            features["imaging_type"] = sample["imaging_type"]
            rows.append(features)

    return pd.DataFrame(rows)


def _clean_colony_label_table(colony_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop colony rows whose geometry contradicts their shape_label (issue #10).

    Returns the cleaned table and removal statistics for `training_meta`.
    Removal counts are emitted for every supported shape so the audit trail is
    complete (shapes without a rule always report 0).
    """
    removals: dict[str, int] = {shape: 0 for shape in SUPPORTED_SHAPES}
    if colony_table.empty:
        stats = {
            "colony_rows_before_cleaning": 0,
            "colony_rows_removed_by_cleaning": 0,
            "colony_cleaning_removals": removals,
        }
        return colony_table.copy(), stats

    keep = pd.Series(True, index=colony_table.index)
    for shape_label, rules in SHAPE_CLEANING_RULES.items():
        contradict = colony_table["shape_label"] == shape_label
        if "max_aspect_ratio" in rules:
            contradict &= colony_table["aspect_ratio"] > rules["max_aspect_ratio"]
        if "min_aspect_ratio" in rules:
            contradict &= colony_table["aspect_ratio"] < rules["min_aspect_ratio"]
        if "max_solidity" in rules:
            contradict &= colony_table["solidity"] > rules["max_solidity"]
        removals[shape_label] = int(contradict.sum())
        keep &= ~contradict

    cleaned = colony_table.loc[keep].copy()
    stats = {
        "colony_rows_before_cleaning": int(len(colony_table)),
        "colony_rows_removed_by_cleaning": int((~keep).sum()),
        "colony_cleaning_removals": removals,
    }
    return cleaned, stats


def _split_colonies_by_image(
    colony_table: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Split colony rows by image so no plate leaks between train and test (issue #6)."""
    if colony_table.empty or "image_id" not in colony_table.columns:
        raise ValueError("colony_table must be non-empty and contain an image_id column.")

    n_images = colony_table["image_id"].nunique()
    n_test_images = int(np.ceil(test_size * n_images))
    n_train_images = n_images - n_test_images
    if n_test_images == 0 or n_train_images == 0:
        raise ValueError(
            f"Cannot build an image-level split: {n_images} images yield "
            f"{n_train_images} train / {n_test_images} test images for test_size={test_size}."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_pos, test_pos = next(splitter.split(colony_table, groups=colony_table["image_id"].to_numpy()))
    return colony_table.index[train_pos], colony_table.index[test_pos]


def _classification_summary(
    y_true: list[str],
    y_pred: list[str],
    scoring_method: str = "accuracy",
    modalities: pd.Series | None = None,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "scoring_method": scoring_method,
        "report": report,
    }
    if scoring_method == "balanced_accuracy":
        summary["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if modalities is not None:
        summary["per_modality_accuracy"] = _per_modality_accuracy(y_true, y_pred, modalities)
    return summary


def _oversample_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Bootstrap minority classes up to the majority count within the training fold.

    Only the training split is resampled here (callers pass the train fold), so the
    test split is never touched — no label leakage.
    """
    counts = y_train.value_counts()
    target = int(counts.max())
    if counts.min() == target:
        return x_train.copy(), y_train.copy()

    rng = np.random.default_rng(random_state)
    parts: list[np.ndarray] = []
    for label, count in counts.items():
        class_idx = y_train[y_train == label].index.to_numpy()
        if count < target:
            extra = rng.choice(class_idx, size=target - count, replace=True)
            parts.append(np.concatenate([class_idx, extra]))
        else:
            parts.append(class_idx)

    row_index = np.concatenate(parts)
    return x_train.loc[row_index].copy(), y_train.loc[row_index].copy()


def _per_modality_accuracy(
    y_true: list[str],
    y_pred: list[str],
    modalities: pd.Series,
) -> dict[str, float]:
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    mod_arr = np.asarray(modalities)
    out: dict[str, float] = {}
    for modality in sorted(set(mod_arr)):
        mask = mod_arr == modality
        out[str(modality)] = float(accuracy_score(true_arr[mask], pred_arr[mask]))
    return out


_SCORING_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
}


def _per_modality_metrics(test_table: pd.DataFrame, y_true_col: str, y_pred: list[str]) -> dict[str, Any]:
    """Accuracy per imaging_type on the held-out test rows (issue #8)."""
    table = test_table.reset_index(drop=True).copy()
    table["_pred"] = list(y_pred)
    result: dict[str, Any] = {}
    for modality in sorted(table["imaging_type"].dropna().unique()):
        sub = table[table["imaging_type"] == modality]
        if len(sub) == 0:
            continue
        result[modality] = {
            "accuracy": float(accuracy_score(sub[y_true_col], sub["_pred"])),
            "n": int(len(sub)),
        }
    return result


def _fit_best_ensemble_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    scoring: str = "accuracy",
    test_modalities: pd.Series | None = None,
    candidates: list[tuple[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any], str]:
    if scoring not in _SCORING_METRICS:
        raise ValueError(f"Unsupported scoring metric: {scoring!r}")
    scoring_fn = _SCORING_METRICS[scoring]

    if candidates is None:
        candidates = [
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=380,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=520,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
            (
                "knn_scaled",
                make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(n_neighbors=5, weights="distance"),
                ),
            ),
        ]

    x_train_res, y_train_res = _oversample_train(x_train, y_train, random_state)

    best_name = ""
    best_model: Any = None
    best_summary: dict[str, Any] | None = None
    best_score = -1.0

    for model_name, model in candidates:
        model.fit(x_train_res, y_train_res)
        pred = model.predict(x_test)
        summary = _classification_summary(
            list(y_test),
            list(pred),
            scoring_method=scoring,
            modalities=test_modalities,
        )
        score = float(scoring_fn(list(y_test), list(pred)))
        if score > best_score:
            best_score = score
            best_name = model_name
            best_model = model
            best_summary = summary

    if best_model is None or best_summary is None:
        raise RuntimeError("Failed to train any candidate model.")

    return best_model, best_summary, best_name


def _train_group_species_models(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    image_feature_cols: list[str],
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    group_models: dict[str, Any] = {}
    group_metrics: dict[str, Any] = {}
    group_model_choices: dict[str, str] = {}

    for group_name in sorted(train_table["taxonomy_group"].unique()):
        train_group = train_table[train_table["taxonomy_group"] == group_name]
        test_group = test_table[test_table["taxonomy_group"] == group_name]
        x_train = train_group[image_feature_cols]
        y_train = train_group["organism"]
        x_test = test_group[image_feature_cols]
        y_test = test_group["organism"]

        if y_train.nunique() < 2:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            group_models[group_name] = model
            if len(y_test) == 0:
                group_metrics[group_name] = {"accuracy": 1.0, "report": {}}
            else:
                pred = model.predict(x_test)
                group_metrics[group_name] = _classification_summary(list(y_test), list(pred))
            group_model_choices[group_name] = "random_forest_single_class"
            continue

        model, summary, model_name = _fit_best_ensemble_model(
            x_train,
            y_train,
            x_test,
            y_test,
            random_state=random_state,
            test_modalities=test_group["imaging_type"],
        )
        group_models[group_name] = model
        group_metrics[group_name] = summary
        group_model_choices[group_name] = model_name

    return group_models, group_metrics, group_model_choices


def train_models(
    dataset_csv: str | Path,
    workspace_root: str | Path,
    model_output_path: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset_csv = Path(dataset_csv)
    workspace_root = Path(workspace_root)
    model_output_path = Path(model_output_path) if model_output_path else workspace_root / MODEL_PATH
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_df = _load_labeled_dataframe(dataset_csv)

    # Single image-level holdout shared by all image-level models. Augmentation is
    # applied to the training fold only so it cannot leak into evaluation.
    train_df, test_df = train_test_split(
        labeled_df,
        test_size=0.2,
        random_state=random_state,
        stratify=labeled_df["organism"],
    )
    balanced_train_df, modality_balance_stats = _balance_modalities_per_species(train_df, random_state)

    train_table = _build_image_feature_table(balanced_train_df, workspace_root, augment=True)
    test_table = _build_image_feature_table(test_df, workspace_root, augment=False)
    if train_table.empty or test_table.empty:
        raise ValueError("Image-level holdout produced an empty train/test split.")

    image_feature_cols = [
        c
        for c in train_table.columns
        if c
        not in {
            "image_path",
            "organism",
            "organism_type",
            "gram_label",
            "shape_label",
            "taxonomy_group",
            "imaging_type",
            "variant_id",
        }
    ]

    gram_train = train_table[train_table["organism_type"] == "bacteria"].copy()
    gram_test = test_table[test_table["organism_type"] == "bacteria"].copy()
    if gram_train.empty or gram_test.empty:
        raise ValueError("No bacterial rows found for Gram classifier training.")

    gram_model, gram_summary, gram_model_name = _fit_best_ensemble_model(
        gram_train[image_feature_cols],
        gram_train["gram_label"],
        gram_test[image_feature_cols],
        gram_test["gram_label"],
        random_state=random_state,
        test_modalities=gram_test["imaging_type"],
    )
    gram_modality = _per_modality_metrics(
        gram_test, "gram_label", list(gram_model.predict(gram_test[image_feature_cols]))
    )

    organism_type_model, organism_type_summary, organism_type_model_name = _fit_best_ensemble_model(
        train_table[image_feature_cols],
        train_table["organism_type"],
        test_table[image_feature_cols],
        test_table["organism_type"],
        random_state=random_state,
        scoring="balanced_accuracy",
        test_modalities=test_table["imaging_type"],
    )
    organism_type_modality = _per_modality_metrics(
        test_table,
        "organism_type",
        list(organism_type_model.predict(test_table[image_feature_cols])),
    )

    group_model, group_summary, group_model_name = _fit_best_ensemble_model(
        train_table[image_feature_cols],
        train_table["taxonomy_group"],
        test_table[image_feature_cols],
        test_table["taxonomy_group"],
        random_state=random_state,
        test_modalities=test_table["imaging_type"],
    )

    organism_model, organism_summary, organism_model_name = _fit_best_ensemble_model(
        train_table[image_feature_cols],
        train_table["organism"],
        test_table[image_feature_cols],
        test_table["organism"],
        random_state=random_state,
        test_modalities=test_table["imaging_type"],
    )
    organism_modality = _per_modality_metrics(
        test_table,
        "organism",
        list(organism_model.predict(test_table[image_feature_cols])),
    )

    group_species_models, group_species_metrics, group_species_model_choices = _train_group_species_models(
        train_table,
        test_table,
        image_feature_cols,
        random_state=random_state,
    )

    colony_table = _build_colony_feature_table(labeled_df, workspace_root)
    colony_table, colony_cleaning_stats = _clean_colony_label_table(colony_table)
    colony_feature_cols = [c for c in colony_table.columns if c not in {"shape_label", "imaging_type", "image_id"}]
    if colony_table.empty:
        raise ValueError("Could not detect colonies in training images.")

    ys = colony_table["shape_label"]

    shape_train_idx, shape_test_idx = _split_colonies_by_image(
        colony_table,
        test_size=0.2,
        random_state=random_state,
    )
    xs_train = colony_table.loc[shape_train_idx, colony_feature_cols]
    ys_train = colony_table.loc[shape_train_idx, "shape_label"]
    xs_test = colony_table.loc[shape_test_idx, colony_feature_cols]
    ys_test = colony_table.loc[shape_test_idx, "shape_label"]

    shape_model, shape_summary, shape_model_name = _fit_best_ensemble_model(
        xs_train,
        ys_train,
        xs_test,
        ys_test,
        random_state=random_state,
        test_modalities=colony_table.loc[xs_test.index, "imaging_type"],
    )

    artifacts = {
        "gram_model": gram_model,
        "organism_type_model": organism_type_model,
        "group_model": group_model,
        "organism_model": organism_model,
        "group_species_models": group_species_models,
        "shape_model": shape_model,
        "image_feature_columns": image_feature_cols,
        "colony_feature_columns": colony_feature_cols,
        "supported_shape_labels": sorted(set(ys)),
        "supported_organisms": sorted(set(train_table["organism"]).union(test_table["organism"])),
        "organism_metadata": ORGANISM_METADATA,
        "feature_version": FEATURE_VERSION,
        "training_meta": {
            "dataset_csv": str(dataset_csv),
            "workspace_root": str(workspace_root),
            "labeled_image_count": int(len(labeled_df)),
            "train_image_count": int(len(train_df)),
            "train_image_count_before_balancing": int(len(train_df)),
            "train_image_count_after_balancing": int(len(balanced_train_df)),
            "modality_balance_stats": modality_balance_stats,
            "test_image_count": int(len(test_df)),
            "augmented_train_rows": int(len(train_table)),
            "augment_per_image": int(AUGMENT_PER_IMAGE),
            "gram_train_samples": int(len(gram_train)),
            "organism_type_train_samples": int(len(train_table)),
            "group_train_samples": int(len(train_table)),
            "organism_train_samples": int(len(train_table)),
            "colony_train_samples": int(len(colony_table)),
            "colony_rows_before_cleaning": int(colony_cleaning_stats["colony_rows_before_cleaning"]),
            "colony_rows_removed_by_cleaning": int(colony_cleaning_stats["colony_rows_removed_by_cleaning"]),
            "colony_cleaning_removals": dict(colony_cleaning_stats["colony_cleaning_removals"]),
            "feature_version": FEATURE_VERSION,
            "image_feature_count": int(len(image_feature_cols)),
        },
        "model_choices": {
            "gram_model": gram_model_name,
            "organism_type_model": organism_type_model_name,
            "group_model": group_model_name,
            "organism_model": organism_model_name,
            "shape_model": shape_model_name,
            "group_species_models": group_species_model_choices,
        },
    }

    joblib.dump(artifacts, model_output_path)

    metrics = {
        "gram_metrics": gram_summary,
        "organism_type_metrics": organism_type_summary,
        "group_metrics": group_summary,
        "group_species_metrics": group_species_metrics,
        "organism_metrics": organism_summary,
        "shape_metrics": shape_summary,
        "per_modality_metrics": {
            "organism_type_model": organism_type_modality,
            "organism_model": organism_modality,
            "gram_model": gram_modality,
        },
        "model_path": str(model_output_path),
        "training_meta": artifacts["training_meta"],
        "model_choices": artifacts["model_choices"],
    }

    metrics_path = model_output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
