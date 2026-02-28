from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from models.classifier import GlaucomaClassifier
from datetime import datetime
import os

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory history list (This resets when server restarts)
# For Stage-II later, we can connect this to MongoDB/MySQL
history = [] 

# Load your trained model
model = GlaucomaClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("models/glaucoma_detector_final.pth", map_location=DEVICE))
model.eval()

def transform_image(file):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(file).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_tensor = transform_image(file).to(DEVICE)
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Formatting the diagnosis
            status = "Glaucoma Detected" if predicted.item() == 1 else "Healthy Eye"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to history
            history.insert(0, {
                "filename": file.filename,
                "result": status,
                "time": timestamp
            })
            result = status

    return render_template('index.html', result=result, history=history)

if __name__ == '__main__':
    app.run(debug=True)