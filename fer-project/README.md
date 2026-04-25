# Advanced Facial Expression Recognition (FER)

EfficientNet-B2 trained on FER2013 + CK+48 + JAFFE for 7-class facial emotion recognition, with a Gradio webcam demo.

## Emotions
`angry` · `disgust` · `fear` · `happy` · `sad` · `surprise` · `neutral`

## Results (achieved on Apple M1 8 GB)

| Dataset | Accuracy | F1 (Macro) |
|---------|----------|------------|
| FER2013 (test) | ~60% | ~0.60 |
| CK+48 | ~60% | ~0.60 |
| JAFFE | weak | — |

JAFFE is weak due to domain gap (Japanese female subjects; model trained on FER2013 crowd-sourced images).

---

## Setup

```bash
pip install torch torchvision
pip install efficientnet-pytorch
pip install opencv-python scikit-learn scikit-image pillow
pip install gradio matplotlib tqdm pyyaml
```

### Datasets
Place datasets under `fer-project/data/raw/`:
```
data/raw/
├── FER2013/          # FER2013 images (train/val/test splits)
├── CK+48/ck+/        # CK+48 images per-class folders
└── jaffe/jaffe/      # JAFFE images
```

---

## Usage

All commands should be run from inside `fer-project/`.

### 1 · EDA
```bash
jupyter notebook notebooks/01_eda_analysis.ipynb
```

### 2 · Data Threshing (optional quality filter)
```bash
jupyter notebook notebooks/02_data_threshing.ipynb
```

### 3 · Train (notebook — iterative)
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 3 · Train (production run)
```bash
python3 main.py train
```

### 4 · Evaluate across datasets
```bash
python3 main.py evaluate --model models/checkpoints/best_model.pth
```

### 5 · Webcam demo
```bash
python3 app.py
# Open http://127.0.0.1:7860
```

---

## Project Structure

```
fer-project/
├── app.py                        # Gradio webcam demo
├── main.py                       # CLI entry point (train / evaluate)
├── config/
│   └── config.yaml               # All hyperparameters and paths
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_data_threshing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── dataset_loader.py     # FER2013 / CK+ / JAFFE dataset classes
│   │   ├── augmentation.py
│   │   └── data_thresher.py
│   ├── models/
│   │   ├── efficientnet_model.py # EfficientNetFER + EnsembleFER
│   │   ├── trainer.py            # FERTrainer (train / validate / checkpoint)
│   │   └── ensemble.py
│   ├── evaluation/
│   │   └── cross_dataset_eval.py
│   └── utils/
│       ├── config.py
│       ├── metrics.py
│       └── visualization.py
├── data/                         # (git-ignored) raw + processed datasets
├── models/                       # (git-ignored) checkpoints
└── outputs/                      # (git-ignored) figures, logs, reports
```

---

## Key Training Config

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B2 |
| Input size | 224 × 224 |
| Batch size | 16 (M1 8 GB safe) |
| Epochs | 20 |
| Optimizer | Adam (lr=0.001) |
| Scheduler | CosineAnnealingLR |
| Backbone freeze | first 5 epochs |
| Dropout | 0.3 |
| Label smoothing | 0.1 |
| Class weighting | inverse-frequency (max 5×) |
| Sampler | WeightedRandomSampler |

---

## Hardware Notes (Apple M1 8 GB)

- Device auto-selected: MPS → CUDA → CPU
- `batch_size: 16` avoids unified-memory OOM
- Training 20 epochs ≈ 2–2.5 hours
- `num_workers: 0` required (MPS + multiprocessing conflict)
