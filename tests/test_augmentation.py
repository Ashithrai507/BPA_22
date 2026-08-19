import pytest
import torch

from bacteria_assistant.dl.model import LabelSmoothingCrossEntropy
from bacteria_assistant.dl.trainer import get_augmentation_transforms, predict_with_tta


def test_augmentation_transforms():
    transform = get_augmentation_transforms()
    assert transform is not None


def test_label_smoothing():
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(2, 10)
    target = torch.tensor([1, 3])
    loss = criterion(pred, target)
    assert loss.item() > 0


def test_predict_with_tta():
    from bacteria_assistant.dl.model import EmbeddingModel

    model = EmbeddingModel(pretrained=False, num_species=5)
    model.eval()
    image_tensor = torch.randn(3, 224, 224)
    result = predict_with_tta(model, image_tensor, n_augmentations=3)
    assert result.shape[0] == 1
    assert result.shape[1] == 5
    assert result.sum().item() == pytest.approx(1.0, abs=0.01)
