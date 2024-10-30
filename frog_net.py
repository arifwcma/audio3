import torch
import torch.nn as nn

class FrogNet(nn.Module):
    def __init__(self):
        super(FrogNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 24, 128)
        self.fc2 = nn.Linear(128, 14)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 24)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
