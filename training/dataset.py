import os
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root):
        self.files = []
        self.labels = []
        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".mp3"):
                        self.files.append(os.path.join(label_path, file))
                        self.labels.append(int(label) - 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.files[idx])
        return waveform.unsqueeze(0), self.labels[idx]
