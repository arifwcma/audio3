import torch.nn as nn
from torchvggish import vggish

class VGGishClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.encoder = vggish(postprocess=False)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
