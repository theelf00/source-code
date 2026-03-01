import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.seed import set_seed

# Path to where you combined the ACRIMA images
IMAGE_DIR = "data/raw"
SEED = 42


def assign_label(filename: str) -> int:
    # ACRIMA naming convention in this project includes "_g_" for glaucoma images.
    return 1 if "_g_" in filename.lower() else 0


def main() -> None:
    set_seed(SEED)

    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    data = [{"filename": f, "label": assign_label(f)} for f in files]

    df = pd.DataFrame(data)

    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv("metadata.csv", index=False)

    print(f"Metadata created with {len(final_df)} images from ACRIMA.")
    print(final_df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
