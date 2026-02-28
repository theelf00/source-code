import io
import os
from datetime import datetime

import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from models.classifier import GlaucomaClassifier

app = Flask(__name__, template_folder="Templates")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# In-memory history list (resets when server restarts)
history = []

model = GlaucomaClassifier(num_classes=2, use_pretrained=False).to(DEVICE)
model.load_state_dict(torch.load("models/glaucoma_detector_final.pth", map_location=DEVICE))
model.eval()


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def transform_image(file_bytes: bytes):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if file is None or file.filename == "":
            error = "Please select an image file."
        elif not allowed_file(file.filename):
            error = "Invalid file type. Please upload PNG, JPG, or JPEG."
        else:
            try:
                file_bytes = file.read()
                img_tensor = transform_image(file_bytes).to(DEVICE)
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)

                status = "Glaucoma Detected" if predicted.item() == 1 else "Healthy Eye"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                history.insert(
                    0,
                    {
                        "filename": file.filename,
                        "result": status,
                        "time": timestamp,
                    },
                )
                result = status
            except UnidentifiedImageError:
                error = "Unable to read image. Please upload a valid image file."
            except Exception:
                error = "Unexpected error during prediction. Please try again."

    return render_template("index.html", result=result, error=error, history=history)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
