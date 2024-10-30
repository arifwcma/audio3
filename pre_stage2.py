import librosa
import numpy as np
import soundfile as sf
import os

input_folder = 'data/stage1'
stage1_folder = 'data/stage2'


def preproc_a_file(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    y, sr = librosa.load(input_file)

    duration = 100 / 1000
    num_samples = int(duration * sr)
    num_segments = len(y) // num_samples

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_segments):
        start = i * num_samples
        end = start + num_samples
        segment = y[start:end]
        output_path = os.path.join(output_folder, f"{i + 1}.mp3")
        sf.write(output_path, segment, sr)

def preproc():
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            file_path = os.path.join(input_folder, filename)
            outputfile = filename.split(".")[0]
            output_path = os.path.join(stage1_folder, outputfile)
            preproc_a_file(file_path, output_path)

    print("Divided")


if __name__ == '__main__':
    preproc()