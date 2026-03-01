import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import GlaucomaClassifier
from utils.dataset_loader import GlaucomaDataset
from utils.metrics_logger import MetricsLogger
from utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 16
BASELINE_MODEL_PATH = "models/glaucoma_detector_baseline.pth"


def evaluate_split(model, loader):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, batch_preds = torch.max(outputs, 1)
            preds.extend(batch_preds.cpu().numpy())
            labels_all.extend(labels.numpy())

    return accuracy_score(labels_all, preds), f1_score(labels_all, preds, zero_division=1)


def train_baseline():
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = GlaucomaDataset("metadata.csv", "data/raw", transform=transform, split="train")
    val_dataset = GlaucomaDataset("metadata.csv", "data/raw", transform=transform, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logger = MetricsLogger(filename="plots/baseline_training_log.csv")

    best_val_f1 = -1.0
    os.makedirs("models", exist_ok=True)

    print("Starting baseline training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_acc, val_f1 = evaluate_split(model, val_loader)

        logger.log(epoch + 1, train_loss, val_acc * 100)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "seed": SEED,
                    "model_type": "baseline",
                },
                BASELINE_MODEL_PATH,
            )

    logger.save()
    print(f"Baseline training complete. Best model saved at {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    train_baseline()
