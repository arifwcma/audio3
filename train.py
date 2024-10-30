import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from frog_net import FrogNet
from audio_dataset import AudioDataset

def train():
    train_folder = 'data/stage3/train'
    dataset = AudioDataset(train_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FrogNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for inputs, labels in dataloader:
            inputs = inputs.squeeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'frog_net.pth')
