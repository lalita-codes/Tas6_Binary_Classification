# Credit Card Fraud Detection (Binary Classification)
This project is a complete Machine Learning pipeline for detecting fraudulent credit card transactions using a Logistic Regression model.
It follows an industry-standard ML project structure including data preprocessing, training, evaluation, and prediction export.

---

## Problem Statement
Credit card transactions are highly imbalanced:
- Most transactions are normal (Class = 0)
- Very few are fraudulent (Class = 1)

The goal is to build a model that can:
- Detect fraud transactions
- Minimize false negatives (missed fraud cases)
- Handle imbalanced data effectively

---

## Dataset
I used the **Kaggle Credit Card Fraud Dataset**:
- 284,807 transactions
- 492 fraud cases
- Highly imbalanced dataset 

---

## Model Configuration
Model selection is controlled using `config/config.yaml`.

---

### Features:
- `V1` to `V28` -> PCA-transformed anonymized features
- `Time` -> Seconds from first transaction
- `Amount` -> Transaction amount
- `Class` -> Target variable (0 = normal, 1 = fraud)


## Installation

### 1. Clone repository
```bash
git clone https://github.com/your-username/Tas6_Binary_Classification.git
cd Tas6_Binary_Classification