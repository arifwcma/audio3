import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from frog_net import FrogNet
from audio_dataset import AudioDataset

def test():
    test_folder = 'data/stage3/test'
    dataset = AudioDataset(test_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = FrogNet()
    model.load_state_dict(torch.load('frog_net.pth', weights_only=True))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.squeeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()
