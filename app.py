import os
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
from preproc import preprocess_audio
from frog_net import VGGishClassifier
from collections import Counter
import re

app = Flask(__name__, static_folder='.')

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

import re

def collect_long_files(folder):
    folder = os.path.abspath(folder)
    result = {}
    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return result

    for f in sorted(os.listdir(folder)):
        if not f.endswith(".mp3"):
            continue
        match = re.match(r"(\d+)", f)
        if match:
            class_id = match.group(1).lstrip("0") or "0"
            result.setdefault(class_id, []).append(
                os.path.relpath(os.path.join(folder, f)).replace("\\", "/")
            )
    return result

def collect_short_files(folder):
    folder = os.path.abspath(folder)
    result = {}
    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return result

    for cls in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, cls)
        if os.path.isdir(class_path):
            result[cls] = [
                os.path.relpath(os.path.join(class_path, f)).replace("\\", "/")
                for f in os.listdir(class_path) if f.endswith(".mp3")
            ]
    return result



@app.route('/')
def index():
    long_files = collect_long_files('data/source')
    short_files = collect_short_files('data/stage3/test')
    return render_template('index.html', long_files=long_files, short_files=short_files, labels=labels)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('.', filename)

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
        predicted_label = f"{labels[majority_class]} ({confidence}% confidence)"

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
