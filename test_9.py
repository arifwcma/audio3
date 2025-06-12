import os
import torch
from torch.utils.data import DataLoader
from audio_dataset import AudioDataset
from frog_net import VGGishClassifier

def test_class_9_only():
    dataset = AudioDataset('data/stage3/test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGGishClassifier(num_classes=14)
    model.load_state_dict(torch.load('model_weights/frog_net.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            if y.item() != 8:
                continue
            out = model(x)
            pred = torch.argmax(out, dim=1).item()
            result = "Correct" if pred == 8 else f"Incorrect (Predicted: {pred + 1})"
            print(result)

if __name__ == '__main__':
    test_class_9_only()
