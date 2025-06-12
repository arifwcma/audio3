import sys
import torch
from collections import Counter
from preproc import preprocess_audio
from frog_net import VGGishClassifier

labels = [
    "1. Bibron's Toadlet > Pseudophryne blbronll",
    "2. Brown Tree Frog > Litoria ewingli",
    "3. Common Eastern Froglet > Crinia signifera",
    "4. Common Spadefoot/Painted Burrowing Frog > Neobatrachus sudelli",
    "5. Eastern Banjo/Pobblebonk Frog > Limnodynastes dumerilii",
    "6. Growling Grass Frog/Southern Bell Frog > Litoria raniformis",
    "7. Mallee Spadefoot Toad/Painted Frog > Neobatrachus pictus",
    "8. Plains Froglet > Crinia parinsignifera",
    "9. Southern Toadlet > Pseudophryne semimarmorata",
    "10. Spotted Marsh/Grass Frog > Limnodynastes tasmaniensis",
    "11. Striped Marsh Frog > Limnodynastes peronii",
    "12. Victorian Smooth Froglet > Geocrinia victoriana",
    "13. Southern Smooth Froglet/Tasmanian Smooth Froglet > Geocrinia laevis",
    "14. Peron's Tree Frog > Litoria peronii",
]


def predict_file(filepath):
    model = VGGishClassifier(num_classes=14)
    model.load_state_dict(torch.load('model_weights/frog_net.pth', map_location='cpu'))
    model.eval()

    inputs = preprocess_audio(filepath)
    if inputs is None:
        print("Error: Invalid or unreadable audio file.")
        return

    with torch.no_grad():
        all_preds = []
        for segment in inputs:
            segment = segment.unsqueeze(0)  # shape: (1, 1, 96, 64)
            output = model(segment)
            pred = torch.argmax(output, dim=1).item()
            all_preds.append(pred)

        counter = Counter(all_preds)
        majority_class, count = counter.most_common(1)[0]
        confidence = round(100 * count / len(all_preds))
        label = labels[majority_class]
        print(f"Prediction: {label} ({confidence}% confidence)")




if __name__ == '__main__':
    filepath = r"C:\Users\m.rahman\audio3\audio3\data\stage3\test\9\3.mp3"
    filepath = r"C:\Users\m.rahman\audio3\audio3\data\source\01 Track 1.mp3"
    predict_file(filepath)
