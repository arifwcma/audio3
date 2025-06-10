import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score, balanced_accuracy_score
from collections import Counter
from audio_dataset import AudioDataset
from frog_net import VGGishClassifier
import matplotlib.pyplot as plt

def count_samples_per_class(folder):
    counts = Counter()
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        if os.path.isdir(path):
            counts[int(label)] = len([f for f in os.listdir(path) if f.endswith('.mp3')])
    return counts

def test():
    dataset = AudioDataset('data/stage3/test')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = VGGishClassifier(num_classes=14)
    model.load_state_dict(torch.load('model_weights/frog_net.pth'))
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.tolist())
            labels.extend(y.tolist())

    oa = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    aa = rc.mean()
    kappa = cohen_kappa_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
    print(f"Average Accuracy (AA): {aa * 100:.2f}%")
    print(f"Balanced Accuracy:     {bal_acc * 100:.2f}%")
    print(f"Cohen’s Kappa:         {kappa:.4f}")
    print("\nPer-class Precision, Recall, F1:")
    for i, (p, r, f) in enumerate(zip(pr, rc, f1), 1):
        print(f"  Class {i:2d}: Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")

    print("\nConfusion Matrix:")
    print(cm)

    counts = count_samples_per_class('data/stage3/test')
    print("\nTest samples per class:")
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count}")

    print("\nPer-class Accuracy:")
    class_acc = {}
    for i in range(len(cm)):
        correct = cm[i][i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0
        class_acc[i+1] = acc
        print(f"  Class {i+1:2d}: Accuracy={acc:.2f}")

    x = list(range(1, 15))
    sample_counts = [counts.get(c, 0) for c in x]
    accuracies = [class_acc.get(c, 0) * 100 for c in x]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - 0.2 for i in x], sample_counts, width=0.4, label='Sample Count')
    ax.bar([i + 0.2 for i in x], accuracies, width=0.4, label='Accuracy')
    ax.set_xlabel('Class')
    ax.set_ylabel('Value')
    ax.set_title('Samples and Accuracy per Class')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results1.png")

if __name__ == '__main__':
    test()
