"""ResNet-50 with a custom classification head for transfer learning."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load pretrained ResNet-50 and replace the final fully-connected layer.
    Args:
        num_classes: Number of target classes.
        freeze_backbone: If True, freeze all layers except the new head.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


if __name__ == "__main__":
    m = build_model(num_classes=10)
    print(m.fc)
    dummy = torch.randn(4, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape: {out.shape}")  # (4, 10)
