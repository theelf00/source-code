import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import GlaucomaClassifier
from utils.checkpoint import load_model_checkpoint
from utils.dataset_loader import GlaucomaDataset
from utils.seed import set_seed

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/glaucoma_detector_final.pth"
CSV_PATH = "metadata.csv"
IMG_DIR = "data/raw"
SEED = 42
METRICS_OUTPUT = "plots/evaluation_metrics.json"


def evaluate():
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = GlaucomaDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform, split="test")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    model, metadata = load_model_checkpoint(model, MODEL_PATH, DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on the holdout test split...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=1)
    rec = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)

    print("\n" + "=" * 30)
    print(" PROJECT EVALUATION METRICS ")
    print("=" * 30)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("-" * 30)

    unique_labels = np.unique(all_labels)
    all_targets = ["Healthy", "Glaucoma"]
    present_targets = [all_targets[i] for i in unique_labels]

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=present_targets))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_targets,
        yticklabels=present_targets,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Glaucoma Detection - Confusion Matrix")

    plt.savefig("plots/evaluation_results.png")
    print("\nVisual Confusion Matrix saved as 'plots/evaluation_results.png'")

    metrics_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        },
        "num_samples": len(all_labels),
        "labels_present": present_targets,
        "model_checkpoint": MODEL_PATH,
        "checkpoint_metadata": metadata,
    }

    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Evaluation metrics JSON saved as '{METRICS_OUTPUT}'")


if __name__ == "__main__":
    evaluate()
