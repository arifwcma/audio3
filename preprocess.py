import os
import torchaudio
import numpy as np
import os


input_folder = 'data/source'
stage1_folder = 'data/stage1'
stage2_folder = 'data/stage2'
stage3_folder = 'data/stage3'

os.makedirs(stage1_folder, exist_ok=True)
os.makedirs(stage2_folder, exist_ok=True)
os.makedirs(stage3_folder, exist_ok=True)



def remove_silence(audio_tensor, threshold=0.01):
    audio_np = audio_tensor.numpy()
    energy = np.sum(np.abs(audio_np), axis=0)
    non_silent_indices = np.where(energy > threshold)[0]

    if len(non_silent_indices) == 0:
        return None
    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1] + 1
    return audio_tensor[:, start_index:end_index]


for filename in os.listdir(input_folder):
    if filename.endswith('.mp3'):
        file_path = os.path.join(input_folder, filename)
        audio_tensor, sample_rate = torchaudio.load(file_path)
        trimmed_audio = remove_silence(audio_tensor)

        if trimmed_audio is not None:
            outputfile = filename.split(" ")[-1]
            output_path = os.path.join(stage1_folder, outputfile)
            torchaudio.save(output_path, trimmed_audio, sample_rate)

print("Audio processing complete. Refined files are saved in the 'data' folder.")
