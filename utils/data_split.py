import os

import pandas as pd
from sklearn.model_selection import train_test_split


def create_data_splits(
    metadata_path: str = "metadata.csv",
    output_dir: str = "data/splits",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> None:
    """Create train/val/test CSV splits from a metadata file."""
    df = pd.read_csv(metadata_path)

    required_cols = {"filename", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Metadata must include columns {required_cols}. Found: {list(df.columns)}")

    stratify_col = df["label"] if df["label"].nunique() > 1 else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col,
    )

    train_val_stratify = train_val_df["label"] if train_val_df["label"].nunique() > 1 else None
    adjusted_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_stratify,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
