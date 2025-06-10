import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from frog_net import VGGishClassifier
from audio_dataset import AudioDataset

def train():
    dataset = AudioDataset('data/stage3/train')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VGGishClassifier(num_classes=14)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(20):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")

    torch.save(model.state_dict(), 'model_weights/frog_net.pth')

if __name__ == '__main__':
    train()
