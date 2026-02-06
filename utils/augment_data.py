import torch
import os
from torchvision.utils import save_image
from models.gan_modules import Generator

def generate_balanced_data(model_path, num_to_generate, output_dir='data/raw/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(model_path))
    gen.eval()

    print(f"Generating {num_to_generate} synthetic Glaucoma images...")
    for i in range(num_to_generate):
        # Using noise to generate a new variation
        noise = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            fake_img = gen(noise)
        
        # Save with a prefix so the metadata script knows it's Glaucoma
        save_image(fake_img, os.path.join(output_dir, f"gen_glaucoma_{i}.png"))
    print("Augmentation complete.")

if __name__ == "__main__":
    # Run this after train_gan.py
    generate_balanced_data("models/generator_trained.pth", num_to_generate=50)