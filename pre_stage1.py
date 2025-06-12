import librosa
import numpy as np
import soundfile as sf
import os

input_folder = 'data/source'
stage1_folder = 'data/stage1'


def preproc_a_file(input_file, output_file):
    y, sr = librosa.load(input_file, sr=16000)
    non_silent_indices = librosa.effects.split(y, top_db=20)
    trimmed_audio = np.concatenate([y[start:end] for start, end in non_silent_indices])
    sf.write(output_file, trimmed_audio, sr)

def preproc():
    os.makedirs(stage1_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            file_path = os.path.join(input_folder, filename)
            outputfile = filename.split(" ")[-1]
            output_path = os.path.join(stage1_folder, outputfile)
            preproc_a_file(file_path, output_path)

    print("Audio processing complete. Refined files are saved in the 'data' folder.")


if __name__ == '__main__':
    preproc()