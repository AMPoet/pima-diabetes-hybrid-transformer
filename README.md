# 🩺 Diabetes Prediction — PCA · SMOTE · XGBoost · Hybrid Transformers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/Conference-Applied%20AI%20Frontiers%202025-orange)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org/)

> **A Multistage PCA–SMOTE Preprocessing Pipeline for Diabetes Prediction Using XGBoost and Hybrid Transformers**
>
> *Amir Mohammad Rahimi · Hedieh Noorian*  
> *1st National Conference on Applied AI Frontiers — University of Science and Technology of Mazandaran, Feb. 9–10, 2025*

> 📌 **Note:** This article has been written by using the **Pima Indians Diabetes Dataset** — a widely-used benchmark comprising 768 clinical records with eight metabolic and hereditary attributes for binary diabetes prediction.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Models](#-models)
- [Results](#-results)
- [Dataset](#-dataset)
- [Installation](https://github.com/AMPoet/pima-diabetes-hybrid-transformer/blob/main/README.md#%EF%B8%8F-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [Authors](#-authors)

---

## 🔍 Overview

This repository is the official implementation of the paper presented at the **1st National Conference on Applied AI Frontiers (2025)**. It provides a complete, reproducible machine learning pipeline for **binary diabetes prediction** on the Pima Indians Diabetes Dataset.

The framework combines three preprocessing stages with a comprehensive model comparison:

| Stage | Method | Purpose |
|---|---|---|
| 1 | **StandardScaler** | Zero mean, unit variance |
| 2 | **PCA** | Remove multicollinearity, orthogonal features |
| 3 | **SMOTE** | Correct class imbalance (train only) |

**Key finding:** The **Transformer + GBDT hybrid** achieves the highest AUC (0.97) and F1 (0.92), outperforming all 13 standalone models.

---

## 🏗 Pipeline Architecture

```
Raw CSV  →  768 samples × 8 clinical features  →  Binary Outcome (0/1)
        │
        ▼
┌──────────────────────────┐
│  Stratified Train/Test   │   75% train  /  25% test
│  Split (stratify=y)      │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  StandardScaler          │   Fit on train → transform both splits
│  (zero-leakage)          │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  PCA (95% variance)      │   Fit on train → transform both splits
│  Removes multicollinearity│
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  SMOTE (train only)      │   Synthesizes minority-class samples
│  Applied AFTER PCA       │   in orthogonal PCA space
└──────────────────────────┘
        │
        ├─────────────────────────┬──────────────────────────┐
        ▼                         ▼                          ▼
 Classical ML              Deep Learning (Keras)     Advanced Models
 LR · SVM · KNN            MLP · CNN · LSTM          SimpleTransformer
 RF · DT                                             TabTransformer
        │                                            FT-Transformer
        ▼                                            GBDT+Transformer
 Gradient Boosting                                   TabNet · DeepGBM
 XGBoost · LightGBM                                  SAINT · NODE
 CatBoost
        │
        └─────────────────────────┴──────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Evaluation (6 Metrics)  │
                    │  Accuracy · Precision        │
                    │  Recall · Specificity        │
                    │  F1 Score · AUC-ROC          │
                    └─────────────────────────────┘
                                  │
                      model_comparison_results.csv
                      advanced_model_results.csv
```

---

## 🤖 Models

| Category | Models |
|---|---|
| **Classical ML** | Logistic Regression (LR), SVM (RBF), KNN, Decision Tree (DT), Random Forest (RF) |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost |
| **Sklearn MLP** | MLPClassifier |
| **Deep Learning — Keras** | MLP, 1D-CNN, LSTM |
| **Transformers — PyTorch** | SimpleTransformer, TabTransformer, FT-Transformer |
| **Advanced Tabular** | TabNet, DeepGBM, SAINT, NODE |
| **Hybrid** | **Transformer + GBDT** (best model — AUC 0.97) |

All models go through the **same PCA → SMOTE → HPO → Evaluate** pipeline for a fair comparison.

---

## 📊 Results

Performance on the Pima Indians Diabetes Dataset (25% held-out test set):

| Model | Accuracy | F1 | Specificity | AUC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.81 | 0.78 | 0.84 | 0.86 |
| Decision Tree | 0.79 | 0.76 | 0.83 | 0.82 |
| KNN | 0.83 | 0.80 | 0.86 | 0.88 |
| Random Forest | 0.87 | 0.85 | 0.89 | 0.92 |
| SVM (RBF) | 0.88 | 0.87 | 0.90 | 0.93 |
| MLP (Keras) | 0.89 | 0.87 | 0.90 | 0.94 |
| TabTransformer | 0.89 | 0.87 | 0.90 | 0.94 |
| Transformer | 0.90 | 0.88 | 0.91 | 0.95 |
| CatBoost | 0.91 | 0.89 | 0.92 | 0.95 |
| LightGBM | 0.91 | 0.89 | 0.92 | 0.95 |
| XGBoost | 0.92 | 0.91 | 0.93 | 0.96 |
| FT-Transformer | 0.92 | 0.90 | 0.93 | 0.96 |
| **Transformer + GBDT** | **0.93** | **0.92** | **0.93** | **0.97** |

> ✅ **Best model:** Transformer + GBDT Hybrid — AUC **0.97**, F1 **0.92**

---

## 🗂 Dataset

> 📌 **This article has been written by using the Pima Indians Diabetes Dataset.**

| Property | Value |
|---|---|
| **Source** | National Institute of Diabetes and Digestive and Kidney Diseases |
| **Samples** | 768 clinical records |
| **Features** | 8 metabolic & hereditary attributes |
| **Target** | Binary — 1 = Diabetic, 0 = Non-diabetic |
| **Class Ratio** | ~35% positive (diabetic) / ~65% negative |

| Feature | Description |
|---|---|
| Pregnancies | Number of prior pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Hereditary diabetes influence score |
| Age | Age in years |

📥 **Download:** [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
Place the CSV at `data/Test_DataSet.csv`.

---

## ⚙️ Installation

### Prerequisites

- Python **3.8+**
- pip 21+

### Clone the repository

```bash
git clone https://github.com/<your-username>/diabetes-pca-smote-transformer.git
cd diabetes-pca-smote-transformer
```

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### Install dependencies

**Core (classical + GBDT models):**
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn xgboost catboost lightgbm
```

**Deep learning — TensorFlow (MLP, CNN, LSTM):**
```bash
pip install tensorflow
```

**Transformers — PyTorch:**
```bash
pip install torch torchvision torchaudio
```

**Advanced tabular (optional):**
```bash
pip install pytorch-tabnet deepgbm pytorch_tabular[all] rtdl
```

---

## 🚀 Usage

### Open the Jupyter Notebook

```bash
jupyter notebook notebooks/Diabetes_PCA_SMOTE_Transformer_Pipeline.ipynb
```

The notebook contains all experiments in order:

| Section | Content |
|---|---|
| Section 1 | Install all libraries |
| Section 2 | Preprocessing: StandardScaler → PCA → SMOTE |
| Section 3 | MLP (Keras) with RandomizedSearchCV |
| Section 4 | LSTM with RandomizedSearchCV |
| Section 5 | Full pipeline: all classical, GBDT & Keras models |
| Section 6 | Transformer architectures (PyTorch) |
| Section 7 | CatBoost & LightGBM standalone |
| Section 8 | Advanced tabular: TabNet, DeepGBM, SAINT, Hybrid |
| Section 9 | Results table & discussion |
| Section 10 | Conclusion & citation |

---

## 📁 Repository Structure

```
diabetes-pca-smote-transformer/
│
├── README.md                                          ← This file
├── LICENSE                                            ← MIT License
├── .gitignore
├── requirements.txt
├── CONTRIBUTING.md
│
├── notebooks/
│   └── Diabetes_PCA_SMOTE_Transformer_Pipeline.ipynb ← Main notebook (all experiments)
│
└── data/
    └── README.md                                      ← Dataset download instructions
```

> All code is contained within the single Jupyter notebook following academic paper convention.

---

## 📄 Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{rahimi2025diabetes,
  title     = {A Multistage PCA–SMOTE Preprocessing Pipeline for Diabetes Prediction
               Using XGBoost and Hybrid Transformers},
  author    = {Rahimi, Amir Mohammad and Noorian, Hedieh},
  booktitle = {1st National Conference on Applied Artificial Intelligence Frontiers},
  address   = {University of Science and Technology of Mazandaran, Iran},
  year      = {2025},
  month     = {February}
}
```

---

## 👥 Authors

| Name | Role | Affiliation | Contact |
|---|---|---|---|
| **Amir Mohammad Rahimi** | First Author | University of Seyed Jamal E Asadabadi, Kermanshah, Iran | [Amirrahimi815@gmail.com](mailto:Amirrahimi815@gmail.com) |
| **[Hedieh Noorian](https://github.com/hediehnoo)** | Corresponding Author ⭐ | Hamedan University of Technology, Hamedan, Iran | [Hediehnoo2020@gmail.com](mailto:Hediehnoo2020@gmail.com) |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for reproducible medical AI research<br>
  <i>Pima Indians Diabetes Dataset · PCA · SMOTE · XGBoost · Hybrid Transformers</i>
</p>
