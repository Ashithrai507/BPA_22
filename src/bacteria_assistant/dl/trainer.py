from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split

from .dataset import ImageDataset
from .losses import CombinedLoss


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.3
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    val_fraction: float = 0.2
    random_state: int = 42
    num_workers: int = 0
    seed: int = 42
    frozen_epochs: int = 0
    use_scheduler: bool = True
    augment: bool = False
    input_size: int = 224


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_dataset(
    dataset: ImageDataset,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[ImageDataset, ImageDataset]:
    """Stratified image-level split on species, mirroring training.py."""
    indices = list(range(len(dataset)))
    species = [dataset[i][1]["species"] for i in indices]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=random_state,
        stratify=species,
    )
    return dataset.subset(train_idx), dataset.subset(val_idx)


def _collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    imgs = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return imgs, targets


def fit_embedding_model(
    model: torch.nn.Module,
    dataset: ImageDataset,
    config: TrainingConfig,
    checkpoint_path: Path,
    device: str | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Train the embedding model and save a checkpoint.

    Returns (history, metadata) where history holds per-epoch train loss and
    validation species accuracy, and metadata summarizes the run.
    """
    _set_seed(config.seed)
    device = device or get_device()

    train_ds, val_ds = split_dataset(dataset, config.val_fraction, config.random_state)
    from .transforms import inference_transform, train_transform

    if config.augment:
        train_ds = train_ds.with_transform(train_transform(seed=config.seed, input_size=config.input_size))
    else:
        train_ds = train_ds.with_transform(inference_transform(config.input_size))
    val_ds = val_ds.with_transform(inference_transform(config.input_size))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
    )

    model = model.to(device)
    if config.frozen_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    criterion = CombinedLoss(
        aux_weight=config.aux_weight,
        contrastive_weight=config.contrastive_weight,
        temperature=config.temperature,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs - config.frozen_epochs, 1))
        if config.use_scheduler
        else None
    )

    history: dict[str, list[float]] = {"loss": [], "val_acc_species": []}
    best_acc = -1.0

    for epoch in range(config.epochs):
        if epoch == config.frozen_epochs and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            optimizer.zero_grad()
            out = model(imgs)
            loss, _ = criterion(out["logits"], targets, out["embeddings"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_acc = _evaluate_species(model, val_loader, device)
        history["loss"].append(train_loss)
        history["val_acc_species"].append(val_acc)
        if scheduler is not None:
            scheduler.step()

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_encoders": {key: enc.to_dict() for key, enc in dataset.encoders.items()},
                    "config": {
                        "backbone": getattr(model, "backbone_name", "custom"),
                        "embedding_dim": getattr(model, "embedding_dim", 0),
                        "num_species": len(dataset.encoders["species"]),
                        "epochs": epoch + 1,
                        "lr": config.lr,
                        "aux_weight": config.aux_weight,
                        "contrastive_weight": config.contrastive_weight,
                        "input_size": config.input_size,
                    },
                    "val_acc_species": val_acc,
                },
                checkpoint_path,
            )

    metadata: dict[str, Any] = {
        "best_val_acc_species": best_acc,
        "device": device,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    return history, metadata


def _evaluate_species(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out["logits"]["species"].argmax(dim=1)
            correct += (preds.cpu() == targets["species"]).sum().item()
            total += imgs.size(0)
    return correct / max(total, 1)


def checkpoint_signature(checkpoint_path: Path) -> str:
    """Deterministic content hash for a saved checkpoint (test helper)."""
    return hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
