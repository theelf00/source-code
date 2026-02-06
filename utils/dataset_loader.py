import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label