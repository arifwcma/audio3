import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, confusion_matrix, balanced_accuracy_score
)
from collections import Counter
from frog_net import FrogNet
from audio_dataset import AudioDataset

def count_samples_per_class(folder):
    class_counts = Counter()
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.mp3')])
            class_counts[int(class_name)] = count
    return class_counts

def test():
    test_folder = 'data/stage3/test'
    dataset = AudioDataset(test_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = FrogNet()
    model.load_state_dict(torch.load('model_weights/frog_net.pth', weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.squeeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    oa = accuracy_score(all_labels, all_preds)
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    aa = class_rec.mean()
    kappa = cohen_kappa_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
    print(f"Average Accuracy (AA): {aa * 100:.2f}%")
    print(f"Balanced Accuracy:     {bal_acc * 100:.2f}%")
    print(f"Cohen’s Kappa:         {kappa:.4f}")
    print("\nPer-class Precision, Recall, F1:")
    for i, (p, r, f) in enumerate(zip(class_prec, class_rec, class_f1), 1):
        print(f"  Class {i:2d}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")

    print("\nConfusion Matrix:")
    print(cm)

    test_counts = count_samples_per_class(test_folder)

    print("\nTest samples per class:")
    for cls, count in sorted(test_counts.items()):
        print(f"  Class {cls}: {count}")

    print("\nPer-class Accuracy:")
    class_acc = {}
    for i in range(len(cm)):
        correct = cm[i][i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0
        class_acc[i+1] = acc
        print(f"  Class {i+1:2d}: Accuracy={acc:.2f}")

    classes = sorted(test_counts.keys())
    sample_counts = [test_counts.get(c, 0) for c in classes]
    accuracies = [class_acc.get(c, 0) * 100 for c in classes]

    x = range(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], sample_counts, width=width, label='Sample Count')
    ax.bar([i + width/2 for i in x], accuracies, width=width, label='Accuracy')

    ax.set_xlabel('Class')
    ax.set_ylabel('Value')
    ax.set_title('Samples and Accuracy per Class')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.legend()
    plt.tight_layout()
    plt.savefig("results1.png")

if __name__ == "__main__":
    test()
