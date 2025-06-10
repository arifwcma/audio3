import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FrogNet
from dataset import AudioDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrogNet().to(device)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "frog_net.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', ''))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', ''))

    args = parser.parse_args()
    train(args)
