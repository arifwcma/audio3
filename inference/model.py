import torch
import torch.nn as nn

class FrogNet(nn.Module):
    def __init__(self):
        super(FrogNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self._flattened_size = 32 * (2205 // 2 // 2)  # After two poolings of stride 2
        self.fc1 = nn.Linear(self._flattened_size, 128)
        self.fc2 = nn.Linear(128, 14)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
