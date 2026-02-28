# Glaucoma Detection with GAN + Incremental Learning

This repository contains an end-to-end pipeline for glaucoma detection from retinal fundus images using:
- metadata generation from ACRIMA filenames,
- train/validation/test split creation,
- GAN training for synthetic image generation,
- incremental classifier training with distillation,
- evaluation with standard metrics,
- a Flask web app for inference.

## Dataset
Source: https://www.kaggle.com/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images?resource=download

Expected layout:
- `data/raw/` -> ACRIMA images

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline Overview
`main.py` runs all major stages in sequence:
1. `prepare_acrima_metadata.py`
2. `train_gan.py`
3. `train_incremental.py`
4. `evaluate.py`

Run full pipeline:
```bash
python main.py
```

## Stage Commands

### 1) Prepare metadata + data splits
Creates:
- `metadata.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

```bash
python prepare_acrima_metadata.py
```

### 2) Train GAN
Uses `data/splits/train.csv` if available, else falls back to `metadata.csv`.

```bash
python train_gan.py
```

### 3) Incremental classifier training
Uses `data/splits/train.csv` if available.
Tries to load baseline checkpoint for distillation from:
- `models/glaucoma_detector_baseline.pth`

Saves final checkpoint to:
- `models/glaucoma_detector_final.pth`

```bash
python train_incremental.py
```

### 4) Evaluate model
Uses `data/splits/test.csv` if available.
Saves confusion matrix plot to:
- `plots/evaluation_results.png`

```bash
python evaluate.py
```

## Run Flask App
```bash
python app.py
```

Open http://127.0.0.1:5000/.

Production note:
- Set `FLASK_DEBUG=true` only for local debugging.
- Uploads are limited to 5 MB and image extensions `png/jpg/jpeg`.

## Reproducibility
Training and evaluation scripts set deterministic seeds via `utils/reproducibility.py`.

## Outputs
- Models: `models/*.pth`
- Metrics log: `plots/training_log.csv`
- Visuals: `plots/*.png`
