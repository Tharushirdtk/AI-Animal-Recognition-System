from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io, json, os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Paths (robust & safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "animal_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")

# --------------------------------------------------
# Italian → English label mapping
# --------------------------------------------------
LABEL_TRANSLATION = {
    "cane": "dog",
    "gatto": "cat",
    "cavallo": "horse",
    "farfalla": "butterfly",
    "elefante": "elephant",
    "mucca": "cow",
    "pecora": "sheep",
    "pollo": "chicken",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

# --------------------------------------------------
# Load model
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# --------------------------------------------------
# Load class labels
# --------------------------------------------------
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"class_labels.json not found at {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → raw label
index_to_label = {int(v): k for k, v in class_indices.items()}

print("✅ Model loaded")
print("✅ Raw labels:", index_to_label)

# --------------------------------------------------
# Prediction route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Load & preprocess image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)

        class_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_index])

        raw_label = index_to_label[class_index]
        prediction = LABEL_TRANSLATION.get(raw_label, raw_label)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                LABEL_TRANSLATION.get(index_to_label[i], index_to_label[i]):
                    round(float(preds[0][i]), 4)
                for i in range(len(preds[0]))
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
