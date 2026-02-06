import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.classifier import GlaucomaClassifier
from utils.dataset_loader import GlaucomaDataset
from utils.losses import distillation_loss
from utils.metrics_logger import MetricsLogger # Added for Chapter 5 objectives

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 20

def train_incremental():
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = GlaucomaDataset(csv_file='metadata.csv', img_dir='data/raw', transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Initialize Models
    old_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    old_model.eval() 
    
    new_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(new_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Logger to save training_log.csv for your report plots
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
            
            # Loss Calculation (Hybrid approach per your report)
            loss_task = criterion(new_outputs, labels)
            loss_dist = distillation_loss(new_outputs, old_outputs, T=2.0)
            total_loss = loss_task + loss_dist
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Calculate accuracy for the logger
            _, predicted = torch.max(new_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(loader)
        
        # Log epoch results to the CSV file
        logger.log(epoch + 1, epoch_loss, epoch_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # 3. Save Final State
    logger.save()
    torch.save(new_model.state_dict(), "models/glaucoma_detector_final.pth")
    print("Incremental Training Complete! Metrics and Model saved.")

if __name__ == "__main__":
    train_incremental()