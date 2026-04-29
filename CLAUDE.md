# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

9-class histopathological image classification project (colon cancer tissue types) using a custom CNN built with PyTorch. Documentation and comments are in Turkish.

**Classes:** Normal, Tumor, Stroma, Lympho, Complex, Debris, Mucosa, Adipose, Background

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify GPU/CUDA availability
python check.py

# Train model
cd src && python train.py

# Evaluate on test set (requires best_model.pt)
cd src && python evaluate.py

# Run inference API (FastAPI, port 8000)
cd src && uvicorn api:app --reload

# Frontend (Next.js, port 3000)
cd frontend && npm install && npm run dev
npm run build    # production build
npm run lint     # ESLint

# Smoke-test dataset loading
cd src && python dataset.py

# Smoke-test model forward pass
cd src && python model.py
```

## Architecture

The project has two layers: a Python ML backend (`src/`) and a Next.js frontend (`frontend/`).

### Backend — `src/`

All five modules must be run from the `src/` directory since they import each other directly (no package structure).

**`dataset.py`** — Data pipeline. Scans class folders, performs 70/15/15 stratified split per class (seeded), applies augmentation for training (horizontal/vertical flip, 90° rotation, color jitter for stain variation), and returns three `DataLoader`s via `get_dataloaders()`. `CLASS_MAP` maps folder names like `"1. Normal"` to integer indices 0–8.

**`model.py`** — Defines `SimpleCancerNet`: 4 `ConvBlock`s (Conv→BN→ReLU→MaxPool, 3→32→64→128→256 channels, 5×5 first kernel then 3×3), global average pooling to 256-dim, then a 2-layer dense head with dropout. Outputs raw logits (no softmax — `CrossEntropyLoss` expects this). Also exports `CLASS_NAMES`, `CANCER_CLASSES` ({1,4}: Tumor, Complex), `NORMAL_CLASSES` ({0,2,3,6}), and `NONCLINICAL_CLASSES` ({5,7,8}).

**`train.py`** — Training loop with class-weighted `CrossEntropyLoss`, Adam (lr=1e-3, weight_decay=1e-4), `ReduceLROnPlateau` (factor=0.5, patience=3), early stopping (patience=7). All hyperparameters are in `CONFIG` dict at the top. Saves best checkpoint keyed on val loss. Note: `checkpoint_dir` and `plots_dir` in `CONFIG` are hardcoded absolute Windows paths — update them if running on a different machine.

**`evaluate.py`** — Loads `best_model.pt`, runs inference on the test split, and saves confusion matrix (raw + normalized), per-class F1 bar chart, and a text classification report to `outputs/plots/`. Uses `num_workers=0` (Windows-safe).

**`api.py`** — FastAPI server exposing `POST /predict` (multipart file upload) and `GET /health`. Loads the model once at startup via lifespan context. Accepts JPG/PNG/TIFF/BMP, returns JSON with `prediction`, `all_probabilities`, and `meta`. CORS is configured for `http://localhost:3000` only.

### Frontend — `frontend/`

Next.js 14 (App Router) with TypeScript and Tailwind CSS. Single page (`app/page.tsx`) with drag-and-drop upload, image preview, and results display. Calls `http://localhost:8000/predict` directly from the browser. Clinical groups (`Kanser Şüphesi`, `Normal Doku`, `Klinik Dışı`, `Belirsiz`) drive the result card color scheme.

## Data Path

Dataset must be placed at:
```
C:\Users\PC\Desktop\archive\9 Class Colon Cancer Histopathological Image\{class_name}\*.jpg
```
Hardcoded in `dataset.py` as `DATASET_PATH`. Folder names must match `CLASS_MAP` keys (e.g. `"1. Normal"`, `"2. Tumor"`, …).

## Key Configuration

- Hyperparameters: `CONFIG` dict in `train.py`
- Normalization stats (dataset-specific): `MEAN=[0.747, 0.540, 0.716]`, `STD=[0.091, 0.137, 0.091]` — defined in both `dataset.py` and `api.py`
- Confidence threshold: `0.70` — defined in both `model.py` and `api.py`

## Checkpoint Format

`best_model.pt` is a `torch.save` dict with keys: `model_state`, `optimizer` (optimizer state dict), `val_loss`, `val_acc`, `epoch`, `config`. Not a bare `state_dict`. The dropout value is recovered from `config["dropout"]` when loading.

## Windows Notes

- `num_workers > 0` in DataLoader requires `multiprocessing.freeze_support()` — already called in `train.py`'s `__main__` block. Set `num_workers=0` for scripts not using freeze_support (e.g. `evaluate.py` already does this).
- Run the API from `src/` so relative imports resolve: `cd src && uvicorn api:app --reload`.
