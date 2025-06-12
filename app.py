import torch
from flask import Flask, request, render_template, jsonify
from preproc import preprocess_audio
from frog_net import VGGishClassifier
from collections import Counter
from audio_dataset import AudioDataset

app = Flask(__name__)
model = VGGishClassifier(num_classes=14)
model.load_state_dict(torch.load('model_weights/frog_net.pth', map_location='cpu'))
model.eval()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = 'temp.mp3'
    file.save(file_path)

    inputs = preprocess_audio(file_path)

    if inputs is None:
        return jsonify({'prediction': "Audio file error"})

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).tolist()

        counter = Counter(predictions)
        majority_class, count = counter.most_common(1)[0]
        confidence = round(100 * count / len(predictions))

        print(f"\nTotal slices: {len(predictions)}")
        print("Prediction counts per class (descending):")
        for cls_id, cls_count in counter.most_common():
            print(f"  Class {cls_id + 1}: {cls_count} times")

        predicted_label = f"{labels[majority_class]} ({confidence}% confidence)"


    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
