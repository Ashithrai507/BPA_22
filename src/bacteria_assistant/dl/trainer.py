from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split

from .dataset import ImageDataset
from .losses import CombinedLoss


def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def predict_with_tta(model, image_tensor, n_augmentations=5):
    """Test-time augmentation: average predictions over augmented versions."""
    model.eval()
    predictions = []

    base_transform = get_augmentation_transforms()

    from torchvision.transforms.functional import to_pil_image

    with torch.no_grad():
        for _ in range(n_augmentations):
            pil_img = to_pil_image(image_tensor)
            augmented = base_transform(pil_img)
            augmented = augmented.unsqueeze(0)
            output = model(augmented)
            predictions.append(torch.softmax(output["logits"]["species"], dim=1))

    avg_predictions = torch.stack(predictions).mean(dim=0)
    return avg_predictions


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


def train_kfold(
    dataset: ImageDataset,
    model: torch.nn.Module,
    config: TrainingConfig,
    n_folds: int = 5,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Train with stratified k-fold cross-validation.

    Returns a list of dicts, one per fold, each containing:
      - fold: fold index
      - train_acc: final training species accuracy
      - val_acc: final validation species accuracy
      - history: per-epoch loss and val_acc_species lists
    """
    device = device or get_device()
    species_labels = dataset.table["organism"].tolist()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    fold_results: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), species_labels)):
        print(f"Fold {fold + 1}/{n_folds}")

        train_subset = dataset.subset(list(train_idx))
        val_subset = dataset.subset(list(val_idx))

        from .transforms import inference_transform, train_transform

        if config.augment:
            train_subset = train_subset.with_transform(
                train_transform(seed=config.seed, input_size=config.input_size)
            )
        else:
            train_subset = train_subset.with_transform(inference_transform(config.input_size))
        val_subset = val_subset.with_transform(inference_transform(config.input_size))

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=_collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=_collate,
        )

        fold_model = type(model)(**{k: v for k, v in _model_init_kwargs(model)})
        fold_model = fold_model.to(device)

        if config.frozen_epochs > 0 and hasattr(fold_model, "freeze_backbone"):
            fold_model.freeze_backbone()

        criterion = CombinedLoss(
            aux_weight=config.aux_weight,
            contrastive_weight=config.contrastive_weight,
            temperature=config.temperature,
        )
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs - config.frozen_epochs, 1))
            if config.use_scheduler
            else None
        )

        history: dict[str, list[float]] = {"loss": [], "val_acc_species": []}

        for epoch in range(config.epochs):
            if epoch == config.frozen_epochs and hasattr(fold_model, "unfreeze_backbone"):
                fold_model.unfreeze_backbone()
            fold_model.train()
            total_loss = 0.0
            n_batches = 0
            for imgs, targets in train_loader:
                imgs = imgs.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                optimizer.zero_grad()
                out = fold_model(imgs)
                loss, _ = criterion(out["logits"], targets, out["embeddings"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            val_acc = _evaluate_species(fold_model, val_loader, device)
            history["loss"].append(train_loss)
            history["val_acc_species"].append(val_acc)
            if scheduler is not None:
                scheduler.step()

            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"fold_{fold}.pt"
                torch.save(fold_model.state_dict(), ckpt_path)

        train_acc = _evaluate_species(fold_model, train_loader, device)
        fold_results.append({
            "fold": fold,
            "train_acc": train_acc,
            "val_acc": history["val_acc_species"][-1],
            "history": history,
        })

    return fold_results


def train_with_fine_tuning(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TrainingConfig,
    device: str | None = None,
    unfreeze_after_epoch: int = 5,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Fine-tune with discriminative learning rates.

    Backbone parameters get a lower learning rate; head parameters get the
    normal learning rate. After ``unfreeze_after_epoch`` the backbone is
    unfrozen (if it was frozen) and all params use the normal LR.

    Returns (model, history).
    """
    device = device or get_device()
    model = model.to(device)

    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    backbone_lr = config.lr * 0.01
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": config.lr},
    ], weight_decay=config.weight_decay)

    criterion = CombinedLoss(
        aux_weight=config.aux_weight,
        contrastive_weight=config.contrastive_weight,
        temperature=config.temperature,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.use_scheduler
        else None
    )

    history: dict[str, list[float]] = {"loss": [], "val_acc_species": []}

    for epoch in range(config.epochs):
        if epoch == unfreeze_after_epoch and hasattr(model, "unfreeze_backbone"):
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

    return model, history


def _model_init_kwargs(model: torch.nn.Module) -> list[tuple[str, Any]]:
    """Extract constructor kwargs from a model for creating fold copies."""
    if hasattr(model, "backbone_name") and hasattr(model, "species_head"):
        return [
            ("backbone_name", model.backbone_name),
            ("pretrained", False),
            ("num_species", model.species_head.out_features),
            ("num_groups", model.group_head.out_features),
            ("num_grams", model.gram_head.out_features),
            ("num_types", model.type_head.out_features),
        ]
    return []


def checkpoint_signature(checkpoint_path: Path) -> str:
    """Deterministic content hash for a saved checkpoint (test helper)."""
    return hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
