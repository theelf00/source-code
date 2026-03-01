import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import GlaucomaClassifier
from utils.checkpoint import load_model_checkpoint
from utils.dataset_loader import GlaucomaDataset
from utils.losses import distillation_loss
from utils.metrics_logger import MetricsLogger
from utils.seed import set_seed

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 20
SEED = 42
BATCH_SIZE = 16
BASELINE_MODEL_PATH = "models/glaucoma_detector_baseline.pth"
OUTPUT_MODEL_PATH = "models/glaucoma_detector_final.pth"
BEST_MODEL_PATH = "models/glaucoma_detector_best.pth"


def evaluate_split(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=1)
    return acc, f1


def train_incremental():
    set_seed(SEED)

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = GlaucomaDataset(csv_file="metadata.csv", img_dir="data/raw", transform=transform, split="train")
    val_dataset = GlaucomaDataset(csv_file="metadata.csv", img_dir="data/raw", transform=transform, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Models
    if not os.path.exists(BASELINE_MODEL_PATH):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {BASELINE_MODEL_PATH}. "
            "Run train_baseline.py first to save a baseline checkpoint."
        )

    old_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    old_model, _ = load_model_checkpoint(old_model, BASELINE_MODEL_PATH, DEVICE)
    old_model.eval()

    new_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    new_model, _ = load_model_checkpoint(new_model, BASELINE_MODEL_PATH, DEVICE)

    optimizer = optim.Adam(new_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logger = MetricsLogger(filename="plots/training_log.csv")

    best_val_f1 = -1.0
    os.makedirs("models", exist_ok=True)

    print("Starting Incremental Learning Phase...")
    for epoch in range(EPOCHS):
        new_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            new_outputs = new_model(images)
            with torch.no_grad():
                old_outputs = old_model(images)

            loss_task = criterion(new_outputs, labels)
            loss_dist = distillation_loss(new_outputs, old_outputs, T=2.0)
            total_loss = loss_task + loss_dist

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            _, predicted = torch.max(new_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)

        val_acc, val_f1 = evaluate_split(new_model, val_loader)
        logger.log(epoch + 1, epoch_loss, epoch_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_acc:.2f}%, Val Accuracy: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": new_model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "seed": SEED,
                    "model_type": "incremental_best",
                },
                BEST_MODEL_PATH,
            )

    # 3. Save Final State
    logger.save()
    torch.save(
        {
            "model_state_dict": new_model.state_dict(),
            "epoch": EPOCHS,
            "seed": SEED,
            "model_type": "incremental_final",
        },
        OUTPUT_MODEL_PATH,
    )
    print(f"Incremental Training Complete! Best model: {BEST_MODEL_PATH}, Final model: {OUTPUT_MODEL_PATH}.")


if __name__ == "__main__":
    train_incremental()
