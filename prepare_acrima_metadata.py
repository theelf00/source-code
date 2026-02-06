import pandas as pd
import os

# Path to where you combined the ACRIMA images
IMAGE_DIR = 'data/raw'
files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

data = []
for f in files:
    # ACRIMA naming convention often includes 'g' for glaucoma or 'n' for normal
    # Or, if you kept them in folders, logic would be based on folder path.
    # Here, we check for 'glaucoma' in the filename.
    label = 1 if 'glaucoma' in f.lower() or f.startswith('g') else 0
    data.append({'filename': f, 'label': label})

df = pd.DataFrame(data)
df.to_csv('metadata.csv', index=False)
print(f"Metadata created with {len(df)} images from ACRIMA.")