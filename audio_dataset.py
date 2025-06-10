import os
import torch
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

        self.target_sr = 16000
        self.min_samples = int(0.96 * self.target_sr)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=400,
            hop_length=160,
            n_mels=64,
            f_min=125,
            f_max=7500
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.files[idx])
        if sr != self.target_sr:
            resample = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resample(audio)
        audio = audio.mean(dim=0, keepdim=True)

        if audio.shape[1] < self.min_samples:
            pad = self.min_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        else:
            audio = audio[:, :self.min_samples]

        mel_spec = self.mel(audio)
        log_mel = torch.log(mel_spec + 1e-6)

        if log_mel.shape[2] < 96:
            pad = 96 - log_mel.shape[2]
            log_mel = torch.nn.functional.pad(log_mel, (0, pad))
        else:
            log_mel = log_mel[:, :, :96]

        return log_mel.transpose(1, 2), self.labels[idx]
