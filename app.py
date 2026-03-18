from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os, base64, io
from PIL import Image

app = Flask(__name__, static_folder='.')
CORS(app)

CLASS_LABELS = ["Aneurysm", "Edema", "Haemorrhage", "Ischemia", "Normal", "Tumor"]
MODEL_PATH = "alexnet_brain_disorder.keras"
MODEL_ID = "1MpmN2dTXOX2cSgrfxZfFrZiwsL3z4yNQ"
model = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded ✅")
    else:
        print("Model already exists ✅")

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import load_model
        download_model()
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print(f"Model loaded: {model.input_shape}")
    return model

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((227, 227))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
@app.route('/neurolens')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        image_bytes = file.read()
        arr = preprocess(image_bytes)
        mdl = get_model()
        preds = mdl.predict(arr)[0]
        results = sorted(
            [{"class": l, "probability": float(p), "percentage": f"{float(p)*100:.1f}%"}
             for l, p in zip(CLASS_LABELS, preds)],
            key=lambda x: x["probability"], reverse=True
        )
        top = results[0]
        img_b64 = base64.b64encode(image_bytes).decode()
        ext = file.content_type or "image/jpeg"
        return jsonify({
            "success": True,
            "diagnosis": top["class"],
            "confidence": top["percentage"],
            "all_predictions": results,
            "image_data": f"data:{ext};base64,{img_b64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting NeuroLens...")
    get_model()
    app.run(host='0.0.0.0', port=5000)
