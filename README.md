# Loan risk prediction — Data Analytics 2025/2026

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

> **University Project** — Data Analytics Course, A.Y. 2025/2026  
> Alma Mater Studiorum — Università di Bologna  
> Authors: Leonardo Vorabbi, Carlotta Nunziati

---

## Overview

This project tackles a **credit risk classification** problem: predicting the **risk grade** of a loan application based on borrower and loan characteristics.

The full machine learning pipeline is implemented — from raw data ingestion and preprocessing, through feature engineering and model selection, to final evaluation. The project explores both classical ML approaches and deep tabular models.

---

## Objective

> **Predict the `grade` column** — the risk grade assigned to a loan — using the features provided in `train.csv`.

The `grade` variable is a multi-class label representing the creditworthiness of a loan, ranging from low-risk to high-risk categories.

---

## Repository structure

```
Data_Analytics_Project/
│
├── data/
│   └── Dataset2526/          # Raw dataset (train.csv)
│
├── docs/                     # Documentation and project reports
│
├── src/                      # Source code and notebooks
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset (`train.csv`) contains approximately **150 raw features** per loan application:

| Feature Type | Examples |
|---|---|
| **Numerical** | Loan amount, interest rate, income levels, financial indicators, ... |
| **Categorical** | Employment category, loan purpose, housing status, ... |
| **Date/Time** | Timestamps related to loan issuance or borrower history |

---

## Getting started

### Prerequisites

Python **3.9+** and `pip` are required. GPU support (CUDA) is recommended for training deep tabular models.

### 1. Clone the repository

```bash
git clone https://github.com/leovora/Data_Analytics_Project.git
cd Data_Analytics_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

⚠️ `test.py` requires **NumPy >= 2.0.0**, but `pytorch-tabular` pins **NumPy == 1.26.4**, causing a dependency conflict. To resolve it, run the following commands **in order** after installing the requirements:

```bash
pip install -r requirements.txt
pip uninstall numpy -y
pip install numpy==2.0.0
```

## Models & approach

The project experiments with a range of models suitable for tabular data:

- **Baseline models** - KNN, SVM
- **Ensemble methods** - Random Forest
- **NN model** - Feed Forward
- **Deep Tabular models** - TabNet, TabTransformer
- **Evaluation metric** - Multi-class classification (accuracy, F1-score, etc.)

---

## Tech stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core programming language |
| **Jupyter Notebook** | Interactive analysis and experimentation |
| **Pandas / NumPy** | Data manipulation and numerical computing |
| **Scikit-learn** | Classical ML models and preprocessing |
| **Matplotlib / Seaborn** | Data visualization |
| **pytorch-tabnet** | Deep tabular model (TabNet) |
| **tab-transformer-pytorch** | Transformer-based tabular model |
| **pytorch_tabular** | Feed Forward NN |
