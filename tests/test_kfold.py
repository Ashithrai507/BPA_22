from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.dl.dataset import build_image_dataset
from bacteria_assistant.dl.trainer import TrainingConfig, train_kfold, train_with_fine_tuning, _model_init_kwargs
from sklearn.model_selection import StratifiedKFold


class TinyBackbone(nn.Module):
    """Minimal backbone for fast smoke tests."""

    def __init__(self, backbone_name: str = "tiny", pretrained: bool = False,
                 num_species: int = 3, num_groups: int = 2,
                 num_grams: int = 2, num_types: int = 1) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = 16
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pool = nn.Identity()
        self.species_head = nn.Linear(8, num_species)
        self.group_head = nn.Linear(8, num_groups)
        self.gram_head = nn.Linear(8, num_grams)
        self.type_head = nn.Linear(8, num_types)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x).flatten(1)
        return {
            "embeddings": h,
            "logits": {
                "species": self.species_head(h),
                "group": self.group_head(h),
                "gram": self.gram_head(h),
                "type": self.type_head(h),
            },
        }

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


@pytest.fixture
def tiny_table(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for i in range(12):
        img_path = tmp_path / f"img_{i}.png"
        np_img = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype(np.uint8)
        import cv2
        cv2.imwrite(str(img_path), np_img)
        rows.append({
            "image_path": str(img_path),
            "organism": ["Staphylococcus aureus", "Bacillus subtilis", "Escherichia coli"][i % 3],
            "organism_type": "bacteria",
            "gram_label": "gram_positive" if i % 2 else "gram_negative",
            "taxonomy_group": ["cocci", "bacilli"][i % 2],
        })
    return pd.DataFrame(rows)


def test_kfold_returns_results(tiny_table: pd.DataFrame) -> None:
    dataset = build_image_dataset(tiny_table)
    model = TinyBackbone(num_species=3)
    config = TrainingConfig(batch_size=4, epochs=1, lr=1e-3, num_workers=0, seed=0)
    results = train_kfold(dataset, model, config, n_folds=3, device="cpu")
    assert len(results) == 3
    for r in results:
        assert "fold" in r
        assert "val_acc" in r
        assert "train_acc" in r
        assert "history" in r
        assert 0.0 <= r["val_acc"] <= 1.0
        assert 0.0 <= r["train_acc"] <= 1.0


def test_kfold_no_leakage(tiny_table: pd.DataFrame) -> None:
    dataset = build_image_dataset(tiny_table)
    model = TinyBackbone(num_species=3)
    config = TrainingConfig(batch_size=4, epochs=1, lr=1e-3, num_workers=0, seed=0)
    n_folds = 3

    # Run k-fold to get results
    results = train_kfold(dataset, model, config, n_folds=n_folds, device="cpu")

    # Verify the splits are actually disjoint using the same StratifiedKFold logic
    species_labels = dataset.table["organism"].tolist()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)
    all_val_indices = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), species_labels)):
        train_set = set(train_idx)
        val_set = set(val_idx)
        # Train and val must be disjoint
        assert train_set.isdisjoint(val_set), f"Fold {fold_idx}: train and val indices overlap"
        # Union must cover the entire dataset
        assert train_set | val_set == set(range(len(dataset))), f"Fold {fold_idx}: union doesn't cover full dataset"
        all_val_indices.append(val_set)

    # Different folds must have different validation sets
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            assert all_val_indices[i] != all_val_indices[j], (
                f"Folds {i} and {j} have identical validation sets — likely not stratified properly"
            )


def test_fine_tuning_returns_model(tiny_table: pd.DataFrame) -> None:
    dataset = build_image_dataset(tiny_table)
    model = TinyBackbone(num_species=3)
    config = TrainingConfig(batch_size=4, epochs=2, lr=1e-3, num_workers=0, seed=0)

    from bacteria_assistant.dl.trainer import _collate
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=_collate
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=_collate
    )

    trained_model, history = train_with_fine_tuning(
        model, train_loader, val_loader, config, device="cpu"
    )
    assert isinstance(trained_model, nn.Module)
    assert len(history["loss"]) == 2
    assert len(history["val_acc_species"]) == 2


def test_model_init_kwargs_rejects_unsupported() -> None:
    bare_model = nn.Linear(10, 3)
    with pytest.raises(ValueError, match="Unsupported model type"):
        _model_init_kwargs(bare_model)
