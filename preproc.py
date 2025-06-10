import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    non_silent = librosa.effects.split(y, top_db=20)
    trimmed = np.concatenate([y[start:end] for start, end in non_silent])

    duration_ms = 100
    num_samples = int((duration_ms / 1000) * sr)
    total_segments = len(trimmed) // num_samples
    segments = []

    for i in range(total_segments):
        start = i * num_samples
        end = start + num_samples
        segment = trimmed[start:end]
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        segments.append(segment_tensor)

    if len(segments) == 0:
        segment = trimmed
        if len(segment) < num_samples:
            segment = np.pad(segment, (0, num_samples - len(segment)))
        else:
            segment = segment[:num_samples]
        segments.append(torch.tensor(segment, dtype=torch.float32).unsqueeze(0))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        f_min=125,
        f_max=7500
    )

    processed = []
    for segment in segments:
        mel = mel_transform(segment)
        log_mel = torch.log(mel + 1e-6)
        if log_mel.shape[2] < 96:
            log_mel = F.pad(log_mel, (0, 96 - log_mel.shape[2]))
        else:
            log_mel = log_mel[:, :, :96]
        processed.append(log_mel.transpose(1, 2).unsqueeze(0))  # [1, 1, 96, 64]

    return torch.cat(processed, dim=0)  # [N, 1, 96, 64]
