import os
import torch
from frog_net import FrogNet
from audio_dataset import AudioDataset

def test():
    test_folder = 'data/stage3/test'
    dataset = AudioDataset(test_folder)

    model = FrogNet()
    model.load_state_dict(torch.load('frog_net.pth'))
    model.eval()

    correct = 0
    total = 0

    for idx in range(len(dataset)):
        inputs, label = dataset[idx]
        inputs = inputs.squeeze(1)
        with torch.no_grad():
            output = model(inputs)
            predicted = torch.argmax(output, dim=1).item()
            if predicted == label:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100
    average_accuracy = (correct / len(dataset)) * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')
    print(f'Average Accuracy: {average_accuracy:.2f}%')
