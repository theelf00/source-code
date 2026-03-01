import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        required_columns = {"filename", "label"}
        missing = required_columns.difference(self.annotations.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        if split is not None:
            if "split" not in self.annotations.columns:
                raise ValueError("CSV is missing 'split' column required for split filtering")
            self.annotations = self.annotations[self.annotations["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.loc[index, "filename"]
        img_path = os.path.join(self.img_dir, img_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.loc[index, "label"])

        if self.transform:
            image = self.transform(image)

        return image, label
