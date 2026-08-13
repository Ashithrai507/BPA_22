from __future__ import annotations

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import inference_transform

INPUT_SIZE = 224


class LabelEncoder:
    """Deterministic index<->label mapping from a sorted list of unique values."""

    def __init__(self, labels: list[str]) -> None:
        self._labels = sorted({str(label) for label in labels})
        self._index = {label: i for i, label in enumerate(self._labels)}

    def encode(self, label: str) -> int:
        return self._index[str(label)]

    def decode(self, index: int) -> str:
        return self._labels[index]

    def __len__(self) -> int:
        return len(self._labels)

    def to_dict(self) -> dict[str, list[str]]:
        return {"labels": self._labels}


class ImageDataset(Dataset):
    """Loads image paths + encoded multi-task targets from a labeled DataFrame."""

    def __init__(
        self,
        table: pd.DataFrame,
        species_encoder: LabelEncoder,
        group_encoder: LabelEncoder,
        gram_encoder: LabelEncoder,
        type_encoder: LabelEncoder,
        transform=None,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.species_encoder = species_encoder
        self.group_encoder = group_encoder
        self.gram_encoder = gram_encoder
        self.type_encoder = type_encoder
        self.transform = transform or inference_transform(INPUT_SIZE)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.table.iloc[index]
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {row['image_path']}")

        tensor = self.transform(image)
        targets = {
            "species": torch.tensor(self.species_encoder.encode(row["organism"])),
            "group": torch.tensor(self.group_encoder.encode(row["taxonomy_group"])),
            "gram": torch.tensor(self.gram_encoder.encode(row["gram_label"])),
            "type": torch.tensor(self.type_encoder.encode(row["organism_type"])),
        }
        return tensor, targets

    def subset(self, indices: list[int]) -> ImageDataset:
        """Return a new dataset restricted to the given row indices."""
        return ImageDataset(
            self.table.iloc[indices].copy(),
            species_encoder=self.species_encoder,
            group_encoder=self.group_encoder,
            gram_encoder=self.gram_encoder,
            type_encoder=self.type_encoder,
            transform=self.transform,
        )

    def with_transform(self, transform) -> ImageDataset:
        """Return a copy of this dataset using a different transform."""
        return ImageDataset(
            self.table.copy(),
            species_encoder=self.species_encoder,
            group_encoder=self.group_encoder,
            gram_encoder=self.gram_encoder,
            type_encoder=self.type_encoder,
            transform=transform,
        )

    @property
    def encoders(self) -> dict[str, LabelEncoder]:
        return {
            "species": self.species_encoder,
            "group": self.group_encoder,
            "gram": self.gram_encoder,
            "type": self.type_encoder,
        }


def build_image_dataset(table: pd.DataFrame) -> ImageDataset:
    encoders = {
        "species": LabelEncoder(table["organism"].tolist()),
        "group": LabelEncoder(table["taxonomy_group"].tolist()),
        "gram": LabelEncoder(table["gram_label"].tolist()),
        "type": LabelEncoder(table["organism_type"].tolist()),
    }
    return ImageDataset(
        table,
        species_encoder=encoders["species"],
        group_encoder=encoders["group"],
        gram_encoder=encoders["gram"],
        type_encoder=encoders["type"],
    )
