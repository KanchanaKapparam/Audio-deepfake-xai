# 🎙️ Audio Deepfake XAI — Combined Documentation

---

# Part 1: Audio Deepfake XAI (Random Forest Baseline)

A lightweight Explainable AI (XAI) system for detecting whether an audio clip contains a **real (bonafide)** or **synthetic (deepfake/spoof)** voice. The project uses **MFCC feature extraction**, **Random Forest classification**, **confidence calibration**, and **SHAP explainability** to provide transparent and reliable predictions.

## 🚀 Features
- Detects real vs. deepfake speech
- MFCC-based feature extraction
- Random Forest classifier with hyperparameter tuning
- Balanced dataset creation for unbiased training
- Confidence calibration using Platt Scaling and Isotonic Regression
- SHAP-based global and local explainability
- Automated PDF report generation with evaluation metrics and visualizations

## 🛠️ Technologies Used
- Python
- Google Colab
- Librosa
- Scikit-learn
- SHAP
- NumPy
- Pandas
- Matplotlib

## 📂 Workflow
```
Raw Audio
    │
    ▼
Protocol Parsing & EDA
    │
    ▼
Audio Preprocessing
    │
    ▼
MFCC Feature Extraction
    │
    ▼
Feature Standardization
    │
    ▼
Random Forest Training
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Confidence Calibration
    │
    ▼
SHAP Explainability
    │
    ▼
Performance Evaluation
    │
    ▼
Automated PDF Report
```

## 📊 Dataset
This project uses the **ASVspoof Logical Access (LA)** dataset for training and evaluation.
> **Note:** The dataset is not included in this repository due to its size. Please obtain it from the official ASVspoof release or an authorized distribution source before running the notebook.

## 📁 Repository Contents
- `audio_xai_detect.ipynb` – Main Google Colab notebook implementing the complete pipeline.
- `README.md` – Project documentation.

## ▶️ How to Run
1. Clone this repository.
2. Download the required ASVspoof dataset.
3. Open `audio_xai_detect.ipynb` in Google Colab.
4. Install the required dependencies.
5. Run the notebook cells in order.

## 📈 Outputs
The project generates:
- Model performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- Confidence calibration metrics (ECE, Brier Score)
- SHAP explainability plots
- Reliability diagrams
- Automated PDF report summarizing results

## 👩‍💻 Author
**Kanchana Kapparam**

---

# Part 2: Explainable Deepfake Audio Detection with Confidence Calibration (Multi-Model Pipeline)

A multi-model pipeline for detecting synthetic (spoofed) speech on the **ASVspoof 2019 Logical Access (LA)** dataset. The project trains four classifiers — Random Forest, CNN+BiLSTM, ResNet18, and Light CNN (LCNN) — then layers on confidence calibration and explainability (SHAP + GradCAM) to make the detections transparent and trustworthy.

## Table of Contents

- [Overview](#overview)
- [Pipeline Phases](#pipeline-phases)
- [Models](#models)
- [Explainability](#explainability)
- [Calibration](#calibration)
- [Dataset](#dataset-1)
- [Requirements](#requirements)
- [Usage](#usage-1)
- [Output Structure](#output-structure)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

Deepfake audio (text-to-speech / voice conversion attacks) poses a real threat to speaker-verification systems. This notebook-based pipeline:

1. Extracts MFCC features from raw `.flac` audio.
2. Trains a classical Random Forest baseline and three deep learning architectures.
3. Calibrates model confidence with Platt Scaling and Isotonic Regression.
4. Explains decisions globally (SHAP beeswarm) and locally (SHAP waterfall, GradCAM heatmaps).
5. Compiles everything into a consolidated PDF report.

---

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| **Phase 1** | Data preparation, stratified sampling, 1D MFCC extraction, Random Forest baseline + hyperparameter tuning |
| **Phase 2** | 2D MFCC extraction, CNN+BiLSTM training |
| **Phase 3** | ResNet18 (single-channel, trained from scratch) |
| **Phase 4** | Light CNN (LCNN) with Max-Feature-Map activations |
| **Phase 5** | Confidence calibration — Platt Scaling & Isotonic Regression; reliability diagrams, ECE & Brier Score |
| **Phase 6** | Explainable AI — SHAP for RF; GradCAM for all DL models |
| **Phase 7** | PDF report generation, comparative ROC curves, Google Drive backup |

---

## Models

### Random Forest (baseline)
- 1D MFCC features: `[mean(40 coefficients), std(40 coefficients)]` → 80-dimensional vector
- `StandardScaler` normalization
- Hyperparameter search over `n_estimators`, `max_depth`, `min_samples_split` via `GridSearchCV`

### CNN + BiLSTM
- Three Conv2D+BN+ReLU+MaxPool blocks compress the frequency axis
- Bidirectional LSTM (2 layers, hidden=128) captures temporal context
- Global average pooling → FC head

### ResNet18
- Standard ResNet18 with the first conv adapted to single-channel input
- Trained from scratch (no pretrained weights)

### Light CNN (LCNN)
- Four Conv+MFM+Pool blocks with Max-Feature-Map activation (halves channels by element-wise max)
- MFM-gated FC layers for classification

All deep learning models share the same training loop with early stopping (patience = 5) and best-checkpoint saving.

---

## Explainability

**SHAP (Random Forest)**
- `shap.TreeExplainer` on a 100-sample validation subset
- Beeswarm plot — global feature importance across all samples
- Waterfall plot — per-sample decision breakdown

**GradCAM (CNN+BiLSTM, ResNet18, LCNN)**
- Hook-based implementation; no external GradCAM library required
- Target layers: `conv_block3` (CNN+BiLSTM), `layer4` (ResNet18), `conv4` (LCNN)
- Heatmap overlaid on the 2D MFCC spectrogram to highlight which spectro-temporal regions drove the spoof decision

---

## Calibration

Post-hoc calibration is applied to all four models on the validation set:

- **Platt Scaling** — logistic regression on raw logit scores
- **Isotonic Regression** — non-parametric monotonic mapping on predicted probabilities

Calibration quality is measured with **Expected Calibration Error (ECE)** and **Brier Score**, visualized via reliability diagrams (2×2 grid).

---

## Dataset

**ASVspoof 2019 Logical Access (LA)**

The notebook expects the dataset mounted at `/content/drive/MyDrive/LA_Subset.zip` (Google Drive). It uses only the training and development partitions with balanced stratified sampling:

| Split | Genuine | Spoof | Total |
|-------|---------|-------|-------|
| Train | 1,500 | 1,500 | 3,000 |
| Val   | 500   | 500   | 1,000 |

Protocol files: `ASVspoof2019.LA.cm.train.trn.txt` and `ASVspoof2019.LA.cm.dev.trl.txt`.

---

## Requirements

```
torch torchvision
librosa soundfile
scikit-learn
shap
pandas numpy matplotlib seaborn
reportlab
joblib
```

Install with:

```bash
pip install torch torchvision librosa soundfile scikit-learn shap pandas numpy matplotlib seaborn reportlab joblib
```

The notebook is designed to run on **Google Colab** with a GPU runtime. A GPU is strongly recommended for the deep learning phases.

---

## Usage

1. Upload `LA_Subset.zip` to your Google Drive under `MyDrive/`.
2. Open `Audio_xai_pipeline.ipynb` in Google Colab.
3. Mount Drive and run all cells top-to-bottom. Each phase is self-contained and saves artifacts before the next phase begins.
4. Outputs are synced to `MyDrive/Colab Notebooks/project_outputs/` at the end of each phase.

Key configuration constants (top of Phase 1):

```python
SAMPLE_RATE    = 16000
N_MFCC         = 40
TRAIN_BONAFIDE_SIZE = 1500
TRAIN_SPOOF_SIZE    = 1500
BATCH_SIZE     = 32
EPOCHS         = 20
LR             = 0.0005
```

---

## Output Structure

```
project_outputs/
├── models/
│   ├── scaler.pkl
│   ├── rf_model.pkl
│   ├── best_rf_model.pkl
│   ├── cnn_bilstm.pth
│   ├── resnet18.pth
│   └── lcnn.pth
├── features/
│   ├── X_train_scaled.npy / y_train.npy
│   ├── X_val_scaled.npy   / y_val.npy
│   ├── X_train_2d.npy     / y_train_2d.npy
│   └── X_val_2d.npy       / y_val_2d.npy
├── plots/
│   ├── class_distribution.png
│   ├── sample_bonafide_signal.png
│   ├── sample_spoof_signal.png
│   ├── random_forest_evaluation.png
│   ├── cnn_bilstm_training.png
│   ├── resnet18_training.png
│   ├── lcnn_training.png
│   ├── calibration_reliability_diagrams.png
│   ├── shap_beeswarm.png
│   ├── shap_waterfall.png
│   ├── cnn_bilstm_gradcam.png
│   ├── resnet18_gradcam.png
│   ├── lcnn_gradcam.png
│   └── overlaid_roc_curves.png
└── reports/
    ├── baseline_model_evaluation_metrics.csv
    ├── best_model_evaluation_metrics.csv
    ├── cnn_bilstm_evaluation_metrics.csv
    ├── resnet18_evaluation_metrics.csv
    ├── lcnn_evaluation_metrics.csv
    ├── compiled_comparative_report.csv
    └── Deepfake_Audio_Detection_Report.pdf
```

---

## Evaluation Metrics

Each model is assessed on: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Equal Error Rate (EER), and calibration metrics (ECE, Brier Score) before and after calibration.

---

## 📐 Project Diagram
Interactive Draw.io architecture and workflow diagrams:
**https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Project.drawio&dark=auto#Uhttps%3A%2F%2Fdrive.google.com%2Fuc%3Fid%3D1vgmd9yX_gic_akj3MGg0IYsc2sM-57aT%26export%3Ddownload**
