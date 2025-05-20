from audio_dataset import AudioDataset

def count_samples():
    train_dataset = AudioDataset('data/stage3/train')
    test_dataset = AudioDataset('data/stage3/test')

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

if __name__ == "__main__":
    count_samples()
