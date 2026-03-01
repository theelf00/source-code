import os

import pandas as pd

from utils.data_split import create_data_splits

# Path to ACRIMA images
IMAGE_DIR = "data/raw"
OUTPUT_METADATA = "metadata.csv"


def infer_label(filename: str) -> int:
    """Infer glaucoma label from ACRIMA naming pattern.

    Returns:
        1 for glaucoma, 0 for healthy.
    """
    lower_name = filename.lower()

    # Common ACRIMA naming includes `_g_` for glaucoma images.
    if "_g_" in lower_name or lower_name.startswith("g") or "glaucoma" in lower_name:
        return 1

    return 0


def main() -> None:
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    data = [{"filename": f, "label": infer_label(f)} for f in files]

    df = pd.DataFrame(data).sort_values("filename").reset_index(drop=True)
    df.to_csv(OUTPUT_METADATA, index=False)

    print(f"Metadata created with {len(df)} images from ACRIMA.")
    print(df["label"].value_counts().rename(index={0: "healthy", 1: "glaucoma"}))

    create_data_splits(metadata_path=OUTPUT_METADATA)
    print("Train/validation/test splits created in data/splits/")


if __name__ == "__main__":
    main()
