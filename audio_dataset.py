import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    target_sr = 16000
    min_samples = int(0.96 * target_sr)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        f_min=125,
        f_max=7500
    )
    
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
        file = self.files[idx]
        log_mel = AudioDataset.preproc_file(file)
        return log_mel, self.labels[idx]

    @staticmethod
    def preproc_file(file):
        audio, sr = torchaudio.load(file)
        return AudioDataset.preproc_audio_data(audio, sr)


    @staticmethod
    def preproc_audio_data(audio, sr):
        if sr != AudioDataset.target_sr:
            resample = torchaudio.transforms.Resample(sr, AudioDataset.target_sr)
            audio = resample(audio)
        audio = audio.mean(dim=0, keepdim=True)

        if audio.shape[1] < AudioDataset.min_samples:
            pad = AudioDataset.min_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        else:
            audio = audio[:, :AudioDataset.min_samples]

        mel_spec = AudioDataset.mel(audio)
        log_mel = torch.log(mel_spec + 1e-6)

        if log_mel.shape[2] < 96:
            pad = 96 - log_mel.shape[2]
            log_mel = torch.nn.functional.pad(log_mel, (0, pad))
        else:
            log_mel = log_mel[:, :, :96]

        log_mel = log_mel.transpose(1, 2)
        return log_mel