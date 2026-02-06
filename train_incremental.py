import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.classifier import GlaucomaClassifier
from utils.dataset_loader import GlaucomaDataset
from utils.losses import distillation_loss

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
    # 'old_model' represents the state of knowledge before the new update
    old_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    old_model.eval() # We don't train this, we only learn FROM it
    
    # 'new_model' is the one we are training now
    new_model = GlaucomaClassifier(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(new_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting Incremental Learning Phase...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass on both models
            new_outputs = new_model(images)
            with torch.no_grad():
                old_outputs = old_model(images)
            
            # Loss 1: Task Loss (How well it detects Glaucoma now)
            loss_task = criterion(new_outputs, labels)
            
            # Loss 2: Distillation Loss (How well it remembers old patterns)
            loss_dist = distillation_loss(new_outputs, old_outputs, T=2.0)
            
            # Combine losses (Total Loss)
            total_loss = loss_task + loss_dist
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(loader):.4f}")

    # 3. Save the final detection model
    torch.save(new_model.state_dict(), "models/glaucoma_detector_final.pth")
    print("Incremental Training Complete! Model saved to models/glaucoma_detector_final.pth")

if __name__ == "__main__":
    train_incremental()