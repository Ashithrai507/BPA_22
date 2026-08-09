from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PATH, ORGANISM_METADATA, normalize_organism_name
from .features import colony_to_feature_dict, extract_colonies, extract_image_features, read_image


def _resolve_image_path(workspace_root: Path, image_path: str) -> Path:
    candidate = workspace_root / image_path
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


def _build_image_feature_table(labeled_df: pd.DataFrame, workspace_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in labeled_df.iterrows():
        image_path = _resolve_image_path(workspace_root, str(sample["image_path"]))
        image = read_image(str(image_path))
        f = extract_image_features(image)
        f.update(
            {
                "image_path": str(image_path),
                "organism": sample["organism"],
                "organism_type": sample["organism_type"],
                "gram_label": sample["gram_label"],
                "shape_label": sample["shape_label"],
                "taxonomy_group": sample["taxonomy_group"],
                "imaging_type": sample["imaging_type"],
            }
        )
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
            rows.append(features)

    return pd.DataFrame(rows)


def _image_level_split(
    labeled_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "shape_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split images (not colonies) into train/test with no image overlap.

    Colonies are expanded later, so splitting here guarantees no single
    image's colonies appear in both the training and test sets.
    """
    y = labeled_df[stratify_col]
    split_kwargs: dict[str, Any] = {"test_size": test_size, "random_state": random_state}
    if int(y.value_counts().min()) >= 2:
        split_kwargs["stratify"] = y

    train_idx, test_idx = train_test_split(labeled_df.index, **split_kwargs)
    return labeled_df.loc[train_idx], labeled_df.loc[test_idx]


def _classification_summary(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": report,
    }


def _fit_best_ensemble_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[Any, dict[str, Any], str]:
    candidates: list[tuple[str, Any]] = [
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

    best_name = ""
    best_model: Any = None
    best_summary: dict[str, Any] | None = None
    best_acc = -1.0

    for model_name, model in candidates:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        summary = _classification_summary(list(y_test), list(pred))
        acc = float(summary["accuracy"])
        if acc > best_acc:
            best_acc = acc
            best_name = model_name
            best_model = model
            best_summary = summary

    if best_model is None or best_summary is None:
        raise RuntimeError("Failed to train any candidate model.")

    return best_model, best_summary, best_name


def _train_group_species_models(
    image_table: pd.DataFrame,
    image_feature_cols: list[str],
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    group_models: dict[str, Any] = {}
    group_metrics: dict[str, Any] = {}
    group_model_choices: dict[str, str] = {}

    for group_name in sorted(image_table["taxonomy_group"].unique()):
        group_df = image_table[image_table["taxonomy_group"] == group_name].copy()
        x_group = group_df[image_feature_cols]
        y_group = group_df["organism"]

        if y_group.nunique() < 2:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
            model.fit(x_group, y_group)
            group_models[group_name] = model
            group_metrics[group_name] = {"accuracy": 1.0, "report": {}}
            group_model_choices[group_name] = "random_forest_single_class"
            continue

        value_counts = y_group.value_counts()
        can_stratify = int(value_counts.min()) >= 2

        if can_stratify:
            x_train, x_test, y_train, y_test = train_test_split(
                x_group,
                y_group,
                test_size=0.2,
                random_state=random_state,
                stratify=y_group,
            )
            model, summary, model_name = _fit_best_ensemble_model(
                x_train,
                y_train,
                x_test,
                y_test,
                random_state=random_state,
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_group,
                y_group,
                test_size=0.2,
                random_state=random_state,
            )
            model, summary, model_name = _fit_best_ensemble_model(
                x_train,
                y_train,
                x_test,
                y_test,
                random_state=random_state,
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

    image_table = _build_image_feature_table(labeled_df, workspace_root)
    image_feature_cols = [
        c
        for c in image_table.columns
        if c
        not in {
            "image_path",
            "organism",
            "organism_type",
            "gram_label",
            "shape_label",
            "taxonomy_group",
            "imaging_type",
        }
    ]

    gram_df = image_table[image_table["organism_type"] == "bacteria"].copy()
    if gram_df.empty:
        raise ValueError("No bacterial rows found for Gram classifier training.")

    xg = gram_df[image_feature_cols]
    yg = gram_df["gram_label"]

    xg_train, xg_test, yg_train, yg_test = train_test_split(
        xg,
        yg,
        test_size=0.2,
        random_state=random_state,
        stratify=yg,
    )

    gram_model, gram_summary, gram_model_name = _fit_best_ensemble_model(
        xg_train,
        yg_train,
        xg_test,
        yg_test,
        random_state=random_state,
    )

    xt = image_table[image_feature_cols]
    yt = image_table["organism_type"]
    xt_train, xt_test, yt_train, yt_test = train_test_split(
        xt,
        yt,
        test_size=0.2,
        random_state=random_state,
        stratify=yt,
    )

    organism_type_model, organism_type_summary, organism_type_model_name = _fit_best_ensemble_model(
        xt_train,
        yt_train,
        xt_test,
        yt_test,
        random_state=random_state,
    )

    xgroup = image_table[image_feature_cols]
    ygroup = image_table["taxonomy_group"]
    xgroup_train, xgroup_test, ygroup_train, ygroup_test = train_test_split(
        xgroup,
        ygroup,
        test_size=0.2,
        random_state=random_state,
        stratify=ygroup,
    )

    group_model, group_summary, group_model_name = _fit_best_ensemble_model(
        xgroup_train,
        ygroup_train,
        xgroup_test,
        ygroup_test,
        random_state=random_state,
    )

    xo = image_table[image_feature_cols]
    yo = image_table["organism"]

    xo_train, xo_test, yo_train, yo_test = train_test_split(
        xo,
        yo,
        test_size=0.2,
        random_state=random_state,
        stratify=yo,
    )

    organism_model, organism_summary, organism_model_name = _fit_best_ensemble_model(
        xo_train,
        yo_train,
        xo_test,
        yo_test,
        random_state=random_state,
    )

    group_species_models, group_species_metrics, group_species_model_choices = _train_group_species_models(
        image_table,
        image_feature_cols,
        random_state=random_state,
    )

    colony_table = _build_colony_feature_table(labeled_df, workspace_root)
    colony_feature_cols = [c for c in colony_table.columns if c != "shape_label"]
    if colony_table.empty:
        raise ValueError("Could not detect colonies in training images.")

    xs = colony_table[colony_feature_cols]
    ys = colony_table["shape_label"]

    xs_train, xs_test, ys_train, ys_test = train_test_split(
        xs,
        ys,
        test_size=0.2,
        random_state=random_state,
        stratify=ys,
    )

    shape_model, shape_summary, shape_model_name = _fit_best_ensemble_model(
        xs_train,
        ys_train,
        xs_test,
        ys_test,
        random_state=random_state,
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
        "supported_organisms": sorted(set(yo)),
        "organism_metadata": ORGANISM_METADATA,
        "training_meta": {
            "dataset_csv": str(dataset_csv),
            "workspace_root": str(workspace_root),
            "labeled_image_count": int(len(labeled_df)),
            "gram_train_samples": int(len(gram_df)),
            "organism_type_train_samples": int(len(image_table)),
            "group_train_samples": int(len(image_table)),
            "organism_train_samples": int(len(image_table)),
            "colony_train_samples": int(len(colony_table)),
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
        "model_path": str(model_output_path),
        "training_meta": artifacts["training_meta"],
        "model_choices": artifacts["model_choices"],
    }

    metrics_path = model_output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
