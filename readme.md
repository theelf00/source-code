# Glaucoma Detection Pipeline (ACRIMA)

This repository trains and serves a glaucoma detector using fundus images from ACRIMA.

Dataset source: https://www.kaggle.com/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images?resource=download

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Data layout

Place dataset images in:

- `data/raw/*.jpg|*.jpeg|*.png`

Then generate metadata (deterministic stratified split: train/val/test):

```bash
python prepare_acrima_metadata.py
```

This creates `metadata.csv` with columns:

- `filename`
- `label` (`0=healthy`, `1=glaucoma`)
- `split` (`train|val|test`)

## 3) Training workflow

### 3.1 Baseline classifier training (required)

```bash
python train_baseline.py
```

Outputs:

- `models/glaucoma_detector_baseline.pth`
- `plots/baseline_training_log.csv`

### 3.2 GAN training (optional synthetic augmentation stage)

```bash
python train_gan.py
```

### 3.3 Incremental classifier training

Requires existing baseline checkpoint:

- `models/glaucoma_detector_baseline.pth`

Then run:

```bash
python train_incremental.py
```

Outputs:

- `models/glaucoma_detector_best.pth` (best validation F1)
- `models/glaucoma_detector_final.pth` (final epoch)
- `plots/training_log.csv`

## 4) Evaluation

Evaluation runs only on the `test` split:

```bash
python evaluate.py
```

Output artifacts:

- `plots/evaluation_results.png`
- `plots/evaluation_metrics.json`

## 5) Web inference app

Start Flask app:

```bash
python app.py
```

Routes:

- `GET /` UI upload page
- `POST /predict` JSON inference endpoint
- `GET /health` healthcheck endpoint

Production-safe defaults:

- `FLASK_DEBUG=false` by default
- configurable with `FLASK_DEBUG=true`
- upload size limit: 8MB
- image type validation (`png/jpg/jpeg`)

Optional env vars:

- `PORT` (default `5000`)
- `FLASK_DEBUG` (`true|false`)

## 6) End-to-end pipeline script

Run all stages in sequence:

```bash
python main.py
```

Order:

1. `prepare_acrima_metadata.py`
2. `train_baseline.py`
3. `train_gan.py`
4. `train_incremental.py`
5. `evaluate.py`

## 7) Running tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
