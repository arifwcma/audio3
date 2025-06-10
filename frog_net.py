import torch
import torch.nn as nn
from torchvggish import vggish

class VGGishClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super(VGGishClassifier, self).__init__()
        self.vggish = vggish()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.vggish(x)
        out = self.classifier(features)
        return out
