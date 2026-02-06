import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from models.classifier import GlaucomaClassifier
from utils.dataset_loader import GlaucomaDataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/glaucoma_detector_final.pth"
CSV_PATH = "metadata.csv"
IMG_DIR = "data/raw"

def evaluate():
    # 1. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    dataset = GlaucomaDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 3. Load Model
    model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    # 4. Inference
    print("Running evaluation on the dataset...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Calculate Metrics with Zero Division Handling
    # Setting zero_division=1 ensures the script doesn't error out if a class is missing
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=1)
    rec = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    print("\n" + "="*30)
    print(" PROJECT EVALUATION METRICS ")
    print("="*30)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("-" * 30)

    # 6. Detailed Classification Report - Dynamic Label Handling
    unique_labels = np.unique(all_labels)
    all_targets = ['Healthy', 'Glaucoma']
    # Filter targets to match only what is actually in the current data
    present_targets = [all_targets[i] for i in unique_labels]
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=present_targets))

    # 7. Generate and Save Confusion Matrix Visual
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    # We use dynamic labels here too to prevent plotting errors
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_targets, 
                yticklabels=present_targets)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Glaucoma Detection - Confusion Matrix')
    
    plt.savefig('plots/evaluation_results.png')
    print("\nVisual Confusion Matrix saved as 'plots/evaluation_results.png'")

if __name__ == "__main__":
    evaluate()