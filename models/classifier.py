import torch.nn as nn
from torchvision import models

class GlaucomaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GlaucomaClassifier, self).__init__()
        # We use ResNet18 as our backbone
        self.network = models.resnet18(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)