# 🎙️ Audio Deepfake XAI

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

## 📐 Project Diagram

Interactive Draw.io architecture and workflow diagrams:

**https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Project.drawio&dark=auto#Uhttps%3A%2F%2Fdrive.google.com%2Fuc%3Fid%3D1vgmd9yX_gic_akj3MGg0IYsc2sM-57aT%26export%3Ddownload**

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
