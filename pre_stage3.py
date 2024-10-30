import os
import shutil
import random

def preproc():
    source_folder = 'data/stage2'
    train_folder = 'data/stage3/train'
    test_folder = 'data/stage3/test'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for class_folder in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_folder)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
            os.makedirs(os.path.join(test_folder, class_folder), exist_ok=True)

            files = [f for f in os.listdir(class_path) if f.endswith('.mp3')]
            random.shuffle(files)

            split_index = int(len(files) * 0.75)
            train_files = files[:split_index]
            test_files = files[split_index:]

            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, class_folder, file))

            for file in test_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, class_folder, file))

    print("Audio files have been randomly split into train and test folders.")

if __name__ == '__main__':
    preproc()