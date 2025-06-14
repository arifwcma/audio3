import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa
from pydub import AudioSegment
from audio_dataset import AudioDataset


def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    non_silent_indices = librosa.effects.split(y, top_db=20)
    trimmed_audio = np.concatenate([y[start:end] for start, end in non_silent_indices])
    print(trimmed_audio.shape)
    duration = 100 / 1000
    num_samples = int(duration * sr)
    num_segments = len(trimmed_audio) // num_samples
    segments = []

    for i in range(num_segments):
        start = i * num_samples
        end = start + num_samples
        segment = trimmed_audio[start:end]
        segments.append(segment)

    processed = []
    for segment in segments:
        segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        mel = AudioDataset.preproc_audio_data(segment)
        processed.append(mel.unsqueeze(0))
    if len(processed) == 0:
        return None

    return torch.cat(processed, dim=0)
