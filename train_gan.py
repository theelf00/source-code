import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.gan_modules import Generator, Discriminator
from utils.dataset_loader import GlaucomaDataset

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-4

def train_gan():
    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Ensure you have created metadata.csv first!
    dataset = GlaucomaDataset(csv_file='metadata.csv', img_dir='data/raw', transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Initialize Models
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print("Starting GAN Training...")
    for epoch in range(EPOCHS):
        for i, (real_imgs, _) in enumerate(loader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # --- Train Discriminator ---
            opt_disc.zero_grad()
            label_real = torch.ones(batch_size, 1).to(DEVICE)
            label_fake = torch.zeros(batch_size, 1).to(DEVICE)

            output_real = disc(real_imgs)
            loss_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, 3, 256, 256).to(DEVICE)
            fake_imgs = gen(noise)
            output_fake = disc(fake_imgs.detach())
            loss_fake = criterion(output_fake, label_fake)

            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            opt_disc.step()

            # --- Train Generator ---
            opt_gen.zero_grad()
            output = disc(fake_imgs)
            loss_gen = criterion(output, label_real) # Generator wants Disc to think it's real
            
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch}/{EPOCHS}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # 3. Save the Generator
    torch.save(gen.state_dict(), "models/generator_trained.pth")
    print("GAN Training Complete. Model saved to models/generator_trained.pth")

if __name__ == "__main__":
    train_gan()