import torch
from flask import Flask, request, render_template, jsonify
import torchaudio
from frog_net import FrogNet

app = Flask(__name__)
model = FrogNet()
model.load_state_dict(torch.load('model_weights/frog_net.pth', map_location='cpu'))
model.eval()

def preprocess_audio(file_path):
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.unsqueeze(0)  # add batch dimension (B, C, L)
    return waveform

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

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

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = 'temp.wav'
    file.save(file_path)

    input_tensor = preprocess_audio(file_path)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = labels[predicted_idx]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
