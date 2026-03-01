import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        required_cols = {"filename", "label"}
        if not required_cols.issubset(self.annotations.columns):
            raise ValueError(
                f"Dataset CSV must contain columns {required_cols}. "
                f"Found columns: {list(self.annotations.columns)}"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_id = row["filename"]
        img_path = os.path.join(self.img_dir, img_id)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
