import os
import tempfile
import unittest

import pandas as pd
from PIL import Image

from utils.dataset_loader import GlaucomaDataset


class TestGlaucomaDataset(unittest.TestCase):
    def test_missing_required_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metadata.csv")
            pd.DataFrame([{"file": "a.jpg", "target": 1}]).to_csv(csv_path, index=False)

            with self.assertRaises(ValueError):
                GlaucomaDataset(csv_path, tmpdir)

    def test_missing_image_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metadata.csv")
            pd.DataFrame([{"filename": "missing.jpg", "label": 0, "split": "test"}]).to_csv(csv_path, index=False)

            dataset = GlaucomaDataset(csv_path, tmpdir, split="test")
            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]

    def test_split_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image1_path = os.path.join(tmpdir, "img1.jpg")
            image2_path = os.path.join(tmpdir, "img2.jpg")
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image1_path)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(image2_path)

            csv_path = os.path.join(tmpdir, "metadata.csv")
            pd.DataFrame(
                [
                    {"filename": "img1.jpg", "label": 0, "split": "train"},
                    {"filename": "img2.jpg", "label": 1, "split": "test"},
                ]
            ).to_csv(csv_path, index=False)

            train_ds = GlaucomaDataset(csv_path, tmpdir, split="train")
            test_ds = GlaucomaDataset(csv_path, tmpdir, split="test")

            self.assertEqual(len(train_ds), 1)
            self.assertEqual(len(test_ds), 1)


if __name__ == "__main__":
    unittest.main()
