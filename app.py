import io
import os
from datetime import datetime

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from models.classifier import GlaucomaClassifier
from utils.checkpoint import load_model_checkpoint

app = Flask(__name__, template_folder="Templates")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
FINAL_MODEL_PATH = "models/glaucoma_detector_final.pth"

# In-memory history list (this resets when server restarts)
history = []

model = GlaucomaClassifier(num_classes=2).to(DEVICE)
model, _ = load_model_checkpoint(model, FINAL_MODEL_PATH, DEVICE)
model.eval()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(file_storage):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    file_storage.stream.seek(0)
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


def infer(file_storage):
    img_tensor = transform_image(file_storage).to(DEVICE)
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

    predicted_idx = predicted.item()
    status = "Glaucoma Detected" if predicted_idx == 1 else "Healthy Eye"
    return status, float(confidence.item()), predicted_idx


@app.route("/", methods=["GET", "POST"])
def upload_file():
    result = None
    error = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "Please select an image file to upload."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload PNG/JPG/JPEG images only."
        else:
            try:
                status, confidence, _ = infer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                history.insert(
                    0,
                    {
                        "filename": file.filename,
                        "result": status,
                        "confidence": round(confidence * 100, 2),
                        "time": timestamp,
                    },
                )
                result = status
            except (UnidentifiedImageError, OSError):
                error = "The uploaded file is not a valid image."

    return render_template("index.html", result=result, confidence=confidence, history=history, error=error)


@app.route("/predict", methods=["POST"])
def predict_json():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Missing uploaded file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        status, confidence, predicted_idx = infer(file)
        return jsonify(
            {
                "prediction": status,
                "predicted_class": predicted_idx,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Invalid image payload"}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)
