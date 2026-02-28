import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import GlaucomaClassifier
from utils.dataset_loader import GlaucomaDataset
from utils.losses import distillation_loss
from utils.metrics_logger import MetricsLogger
from utils.reproducibility import set_seed

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 16
SEED = 42
TRAIN_SPLIT_PATH = "data/splits/train.csv"
LEGACY_METADATA_PATH = "metadata.csv"
OLD_MODEL_PATH = "models/glaucoma_detector_baseline.pth"
NEW_MODEL_PATH = "models/glaucoma_detector_final.pth"


def train_incremental():
    set_seed(SEED)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    csv_path = TRAIN_SPLIT_PATH if os.path.exists(TRAIN_SPLIT_PATH) else LEGACY_METADATA_PATH
    dataset = GlaucomaDataset(csv_file=csv_path, img_dir="data/raw", transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    old_model = GlaucomaClassifier(num_classes=2, use_pretrained=False).to(DEVICE)
    if os.path.exists(OLD_MODEL_PATH):
        old_model.load_state_dict(torch.load(OLD_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded old checkpoint for distillation: {OLD_MODEL_PATH}")
    else:
        print(f"Warning: {OLD_MODEL_PATH} not found. Distilling from randomly initialized old model.")
    old_model.eval()

    new_model = GlaucomaClassifier(num_classes=2, use_pretrained=False).to(DEVICE)
    optimizer = optim.Adam(new_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logger = MetricsLogger(filename="plots/training_log.csv")

    print("Starting Incremental Learning Phase...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
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
        epoch_loss = running_loss / len(loader)

        logger.log(epoch + 1, epoch_loss, epoch_acc)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    logger.save()
    torch.save(new_model.state_dict(), NEW_MODEL_PATH)
    print("Incremental Training Complete! Metrics and model saved.")


if __name__ == "__main__":
    train_incremental()
