from __future__ import annotations

import pytest
import torch
from torchvision import transforms

from bacteria_assistant.dl.model import LabelSmoothingCrossEntropy
from bacteria_assistant.dl.trainer import get_augmentation_transforms, get_tta_transforms, predict_with_tta


def test_augmentation_transforms():
    transform = get_augmentation_transforms()
    assert isinstance(transform, transforms.Compose)


def test_tta_transforms():
    transform = get_tta_transforms()
    assert isinstance(transform, transforms.Compose)


def test_label_smoothing():
    pred = torch.randn(2, 10)
    target = torch.tensor([1, 3])

    loss_no_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)(pred, target)
    loss_smooth = LabelSmoothingCrossEntropy(smoothing=0.5)(pred, target)
    assert loss_no_smooth.item() != loss_smooth.item()

    nll_loss = torch.nn.functional.cross_entropy(pred, target)
    assert loss_no_smooth.item() == pytest.approx(nll_loss.item(), abs=1e-5)


def test_predict_with_tta():
    from bacteria_assistant.dl.model import EmbeddingModel

    model = EmbeddingModel(pretrained=False, num_species=5)
    image_tensor = torch.randn(3, 224, 224)
    result = predict_with_tta(model, image_tensor, n_augmentations=3)
    assert result.shape[0] == 1
    assert result.shape[1] == 5
    assert result.sum().item() == pytest.approx(1.0, abs=0.01)
