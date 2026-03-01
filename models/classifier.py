import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class GlaucomaClassifier(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=False):
        super().__init__()
        # Use explicit weights API for torchvision compatibility.
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.network = models.resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)
