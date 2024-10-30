import os
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, folder):
        self.files = []
        self.labels = []
        for label in os.listdir(folder):
            class_folder = os.path.join(folder, label)
            if os.path.isdir(class_folder):
                for file in os.listdir(class_folder):
                    if file.endswith('.mp3'):
                        self.files.append(os.path.join(class_folder, file))
                        self.labels.append(int(label) - 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.files[idx])
        return audio.unsqueeze(0), self.labels[idx]
