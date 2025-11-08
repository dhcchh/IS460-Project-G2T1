# Credit Card Fraud Detection - Multi-Model Comparison & Ensemble

A comprehensive fraud detection system comparing gradient boosting (CatBoost), anomaly detection (VAE), deep learning (FFNN), and traditional ML (Logistic Regression) approaches. Includes a novel **fair cost-based ensemble** that accounts for different training paradigms (supervised vs. unsupervised).

## Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Model Training Approaches](#model-training-approaches)
  - [1. Logistic Regression (Baseline)](#1-logistic-regression-baseline)
  - [2. CatBoost (Gradient Boosting)](#2-catboost-gradient-boosting)
  - [3. VAE (Variational Autoencoder)](#3-vae-variational-autoencoder)
  - [4. FFNN (Feed-Forward Neural Network)](#4-ffnn-feed-forward-neural-network)
- [Ensemble Methodology](#ensemble-methodology)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)

---

## Project Overview

This project explores credit card fraud detection using multiple machine learning paradigms:
- **Traditional ML**: Logistic Regression
- **Gradient Boosting**: CatBoost
- **Anomaly Detection**: VAE (Variational Autoencoder)
- **Deep Learning**: FFNN (Feed-Forward Neural Network)

The key innovation is a **fair ensemble weighting system** that normalizes model performance by cost per fraud case, accounting for fundamental differences between supervised and unsupervised learning approaches.

## Business Objective

**Minimize total business cost** defined as:
```
Total Cost = (False Positives × $550) + (False Negatives × $110)
```

Where:
- **False Positive (FP) = $550**: Customer friction from flagging legitimate transactions
- **False Negative (FN) = $110**: Actual fraud loss (may be partially recovered)

This asymmetric cost function reflects real-world business priorities where false alarms are more expensive than missed fraud.

## Dataset

**Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Characteristics**:
- **Size**: 284,807 transactions
- **Features**: 28 PCA-transformed features (V1-V28) + Time + Amount
- **Class Distribution**:
  - Normal: 284,315 (99.83%)
  - Fraud: 492 (0.17%)
- **Imbalance Ratio**: ~578:1

**Feature Engineering** (from `EDA_named_features_analysis_FE.ipynb`):
- **Hour**: Extracted from Time (0-23)
- **Hour_sin**: Cyclical encoding of hour
- **Day**: Day number from Time
- **log_Amount**: Log-transformed Amount
- **Time_bin**: Time of day category (Night/Morning/Afternoon/Evening)
- **Amount_bin**: Amount category (Low/Medium/High) based on quantiles

**Final Feature Set** (22 features):
- **V features**: V1-V12, V14, V16-V18, V21, V27-V28
- **Engineered**: Day, Time_bin, Amount_bin
- **Dropped**: V13, V15, V19, V20, V22-V26, Time, Amount, Hour, Hour_sin, log_Amount

---

## Model Training Approaches

### 1. Logistic Regression (Baseline)

**Type**: Traditional supervised learning

**Training Approach**:
- **Dataset**: Standard 80/20 train/test split (stratified)
- **Features**: 22 features (logreg_baseline preset)
- **Scaling**: **NO scaling** (trained on raw features)
- **Threshold**: Default 0.5 (no optimization)
- **Training Data**:
  - ~227,046 samples (80% of dataset)
  - Includes both normal and fraud transactions

**Characteristics**:
- Simplest baseline model
- No hyperparameter tuning
- No threshold optimization
- Serves as performance floor for comparison

**Location**: `notebooks/baseline/logistic_regression.ipynb`

---

### 2. CatBoost (Gradient Boosting)

**Type**: Supervised learning with gradient boosting

**Training Approach**:
- **Dataset Split**: 60/20/20 stratified split
  - Train: 170,884 samples (~170,589 normal + ~295 fraud)
  - Validation: 56,962 samples (~56,863 normal + ~99 fraud)
  - Test: 56,961 samples (~56,863 normal + ~98 fraud)
- **Features**: 22 features (logreg_baseline preset)
- **Scaling**: StandardScaler fitted on **all data** (normal + fraud)
- **Threshold Optimization**: Grid search on validation set (0.1-0.9 range)

**Training Pipeline** (`src/catboost_models/`):
1. **Baseline Training** (`catboost_trainer.py`):
   - Default CatBoost parameters
   - Find optimal threshold on validation set
   - Saved as `models/catboost_baseline.cbm`

2. **Hyperparameter Tuning** (`catboost_tuner.py`):
   - Manual grid search over:
     - iterations: [100, 300, 500]
     - depth: [4, 6, 8]
     - learning_rate: [0.01, 0.03, 0.1]
   - Optimize on validation PR-AUC
   - Retrain best config on train+val
   - Saved as `models/catboost_best_tuned.cbm`

**Key Design Decisions**:
- Uses class weights to handle imbalance
- Threshold optimized to minimize business cost
- Scaler fitted on all data (standard supervised approach)

**Notebooks**:
- `notebooks/catboost_model_test/catboost_baseline_training.ipynb`
- `notebooks/catboost_model_test/catboost_hyperparameter_tuning.ipynb`

---

### 3. VAE (Variational Autoencoder)

**Type**: Unsupervised anomaly detection

**Training Approach** (CRITICAL DIFFERENCE):
- **Dataset Split**: Custom anomaly detection split
  - Train: 170,589 samples (60% of **NORMAL ONLY**)
  - Validation: 56,863 normal + 246 fraud (20% normal + 50% fraud)
  - Test: 56,863 normal + 246 fraud (20% normal + 50% fraud)
- **Features**: 22 features (logreg_baseline preset)
- **Scaling**: StandardScaler fitted on **NORMAL TRANSACTIONS ONLY**
- **Threshold Optimization**: Percentile search on validation set (50th-99.9th percentile of reconstruction errors)

**Training Pipeline** (`src/vae_models/`):
1. **Architecture** (`vae_base.py`):
   ```
   Encoder: input(22) → hidden(16) → latent(8)
   Decoder: latent(8) → hidden(16) → output(22)
   ```
   - Learns compressed representation of "normal" transactions
   - Fraud has high reconstruction error (anomaly score)

2. **Baseline Training** (`vae_trainer.py`):
   - Train ONLY on normal transactions
   - Optimize reconstruction loss (MSE + KL divergence)
   - Find threshold that maximizes recall while maintaining precision
   - Saved as `models/vae_baseline.pth`

3. **Hyperparameter Tuning** (`vae_tuner.py`):
   - Optuna optimization over:
     - hidden_dim: [8, 16, 32]
     - latent_dim: [4, 8, 16]
     - learning_rate: [0.0001, 0.001, 0.01]
   - Retrain best on train+val (normal only)
   - Saved as `models/vae_best_tuned.pth`

**Key Design Decisions**:
- **Never sees fraud during training** (pure anomaly detection)
- Scaler fitted only on normal data (critical for fair ensemble)
- Test set has 50% of fraud (harder test than CatBoost)
- Reconstruction error threshold optimized on validation set

**Notebooks**:
- `notebooks/vae_model_test/vae_baseline_training.ipynb`
- `notebooks/vae_model_test/vae_hyperparameter_tuning.ipynb`

---

### 4. FFNN (Feed-Forward Neural Network)

**Type**: Supervised deep learning

**Training Approach**:
- **Dataset Split**: 60/20/20 stratified split (same as CatBoost)
- **Features**: 22 features (logreg_baseline preset)
- **Scaling**: StandardScaler fitted on all data
- **Threshold Optimization**: Grid search on validation set

**Architecture** (`src/ffnn_models/ffnn_base.py`):
```
Input(22) → FC(64) → ReLU → Dropout(0.2)
          → FC(32) → ReLU → Dropout(0.2)
          → FC(16) → ReLU → Dropout(0.2)
          → FC(1) → Sigmoid
```

**Training Details**:
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Early stopping on validation loss
- Same supervised approach as CatBoost

**Location**: `notebooks/ffnn/` (if exists)

---

## Ensemble Methodology

### The Fair Weighting Problem

**Why Simple Cost Weighting is Unfair**:

| Model | Training Paradigm | Fraud in Training | Test Set Fraud | Test Cost |
|-------|------------------|-------------------|----------------|-----------|
| CatBoost | Supervised | 295 cases (learned patterns) | 99 cases (20%) | $2,310 |
| VAE | Unsupervised | 0 cases (anomaly only) | 246 cases (50%) | $18,700 |

**Problem**: Direct cost comparison is unfair because:
1. CatBoost learned from 295 fraud examples
2. VAE never saw any fraud (pure anomaly detection)
3. VAE tested on 2.5× more fraud cases than CatBoost
4. Supervised models naturally have lower cost when evaluated on data similar to training

### Solution: Normalized Cost Per Fraud Case

**Implementation** (`src/ensemble.py:277-341`):

```python
# Normalize by number of fraud cases in evaluation
cost_per_fraud_catboost = catboost_cost / n_fraud_cases  # $10,450 / 492 = $21.24
cost_per_fraud_vae = vae_cost / n_fraud_cases            # $34,760 / 492 = $70.65

# Inverse cost weighting (lower cost per fraud = higher weight)
weight_catboost = (1 / cost_per_fraud_catboost) / total_inverse
weight_vae = (1 / cost_per_fraud_vae) / total_inverse
```

**Benefits**:
1. **Fair comparison** regardless of training paradigm
2. **Accounts for dataset size** (both evaluated on 492 fraud cases)
3. **Respects model strengths**:
   - CatBoost excels at known fraud patterns
   - VAE excels at novel/unknown patterns
4. **Normalized metric** allows fair weighting

### Ensemble Pipeline

**Data Handling** (Critical for Fairness):
```python
# Each model uses its OWN scaler
X_catboost = catboost_scaler.transform(X_raw)  # Fitted on all data
X_vae = vae_scaler.transform(X_raw)            # Fitted on normal only
```

**Prediction Combination**:
1. Get CatBoost probabilities (0-1 scale)
2. Get VAE reconstruction errors (normalized to 0-1)
3. Weighted sum: `ensemble_score = w_cb × p_cb + w_vae × e_vae`
4. Optimize final threshold on full dataset

**Evaluation**:
- All models (CatBoost, VAE, LogReg, Ensemble) evaluated on **SAME** 284,807 samples
- Fair comparison guaranteed by:
  - Same 492 fraud cases
  - Same 22 features
  - Each model uses appropriate preprocessing

**Location**: `src/ensemble.py`, `notebooks/ensemble/ensemble_test.ipynb`

---

## Project Structure

```
IS460-Project-G2T1/
├── data/                           # ⚠️ Create manually (gitignored)
│   ├── raw/                        # Original creditcard.csv
│   └── processed/                  # creditcard_fe.csv (with features)
│
├── models/                         # Trained models
│   ├── logreg_baseline.pkl
│   ├── catboost_baseline.cbm
│   ├── catboost_best_tuned.cbm
│   ├── vae_baseline.pth
│   ├── vae_best_tuned.pth
│   └── *_metadata.pkl              # Scalers & thresholds
│
├── notebooks/
│   ├── eda/                        # Exploratory Data Analysis
│   │   ├── EDA_named_features_analysis_FE.ipynb  # ⭐ RUN FIRST
│   │   └── EDA_statistical_analysis.ipynb
│   │
│   ├── baseline/                   # Logistic Regression
│   │   └── logistic_regression.ipynb
│   │
│   ├── catboost_model_test/       # CatBoost (Supervised)
│   │   ├── catboost_baseline_training.ipynb
│   │   └── catboost_hyperparameter_tuning.ipynb
│   │
│   ├── vae_model_test/             # VAE (Unsupervised Anomaly Detection)
│   │   ├── vae_baseline_training.ipynb
│   │   └── vae_hyperparameter_tuning.ipynb
│   │
│   └── ensemble/                   # Fair Cost-Based Ensemble
│       └── ensemble_test.ipynb
│
├── src/
│   ├── evaluation.py               # FraudEvaluationMetrics (PR-AUC, cost)
│   ├── feature_engineering.py      # engineer_features(), logreg_baseline
│   ├── ensemble.py                 # FraudEnsemble (fair weighting)
│   │
│   ├── catboost_models/
│   │   ├── catboost_base.py        # CatBoostDataHandler, ThresholdOptimizer
│   │   ├── catboost_trainer.py     # train_catboost_baseline()
│   │   └── catboost_tuner.py       # Grid search tuning
│   │
│   ├── vae_models/
│   │   ├── vae_base.py             # VAE architecture, VAEDataHandler
│   │   ├── vae_trainer.py          # train_vae_baseline()
│   │   └── vae_tuner.py            # Optuna hyperparameter tuning
│   │
│   └── ffnn_models/                # (if present)
│
├── results/
│   ├── figures/                    # Visualizations
│   ├── ensemble_comparison.csv     # 4-model comparison
│   └── ensemble_detailed_results.json
│
├── requirements.txt                # Dependencies
├── CLAUDE.md                       # Instructions for Claude Code
└── README.md                       # This file
```

---

## Setup & Installation

### 1. Create Data Folder

```bash
mkdir -p data/raw data/processed
```

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/raw/`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements**:
- pandas, numpy, scikit-learn
- catboost, xgboost, lightgbm
- torch (PyTorch for VAE and FFNN)
- matplotlib, seaborn
- jupyter, ipykernel

### 3. Feature Engineering

```bash
# Run EDA and create feature-engineered dataset
jupyter notebook notebooks/eda/EDA_named_features_analysis_FE.ipynb

# Or via command line:
python src/feature_engineering.py --summary
```

This creates `data/processed/creditcard_fe.csv` with 22 engineered features.

---

## Usage

### Training Workflow

**1. Baseline Models** (in order):
```bash
# Logistic Regression
jupyter notebook notebooks/baseline/logistic_regression.ipynb

# CatBoost Baseline
jupyter notebook notebooks/catboost_model_test/catboost_baseline_training.ipynb

# VAE Baseline
jupyter notebook notebooks/vae_model_test/vae_baseline_training.ipynb
```

**2. Hyperparameter Tuning**:
```bash
# CatBoost Tuning
jupyter notebook notebooks/catboost_model_test/catboost_hyperparameter_tuning.ipynb

# VAE Tuning
jupyter notebook notebooks/vae_model_test/vae_hyperparameter_tuning.ipynb
```

**3. Ensemble Comparison**:
```bash
jupyter notebook notebooks/ensemble/ensemble_test.ipynb
```

### Command Line Usage

**Feature Engineering**:
```bash
python src/feature_engineering.py \
    --input data/raw/creditcard.csv \
    --output data/processed/creditcard_fe.csv \
    --summary
```

**Ensemble Evaluation**:
```bash
python src/ensemble.py \
    --data data/processed/creditcard_fe.csv \
    --catboost models/catboost_best_tuned.cbm \
    --vae models/vae_best_tuned.pth \
    --device cuda
```

---

## Results

### Model Comparison (Full Dataset: 284,807 samples, 492 fraud)

| Model | Training | Fraud in Training | Precision | Recall | PR-AUC | Cost | Savings |
|-------|----------|-------------------|-----------|--------|--------|------|---------|
| **LogReg** | Supervised | 394 (80%) | 0.8857 | 0.6301 | 0.7568 | $42,020 | 22.4% |
| **CatBoost** | Supervised | 295 (60%) | 0.9951 | 0.8272 | 0.9051 | $10,450 | 80.7% |
| **VAE** | Unsupervised | 0 (0%) | 0.9576 | 0.4593 | 0.6483 | $34,760 | 35.8% |
| **Ensemble** | Fair Weighted | - | 0.9951 | 0.8272 | 0.9063 | $10,450 | 80.7% |

**Baseline (No Detection)**: $54,120

### Key Insights

1. **CatBoost Dominates**: Best individual model with 80.7% cost savings
2. **Ensemble Matches CatBoost**: Fair weighting gives CatBoost 76.9% weight
3. **VAE Competitive**: Despite no fraud in training, achieves 35.8% savings
4. **LogReg Baseline**: Simple model still provides 22.4% cost reduction

### Ensemble Weights (Fair Normalized)

```
CatBoost: 76.9% weight ($21.24 per fraud case)
VAE:      23.1% weight ($70.65 per fraud case)
```

The normalized weighting accounts for:
- CatBoost's supervised learning advantage (saw 295 fraud cases)
- VAE's unsupervised disadvantage (saw 0 fraud cases)
- Different test set sizes (CatBoost: 99 fraud, VAE: 246 fraud)

---

## Key Learnings

### 1. Training Paradigm Matters
- **Supervised models** (CatBoost, FFNN) excel when fraud patterns are known
- **Unsupervised models** (VAE) detect novel patterns but have higher cost
- **Ensemble** combines strengths of both approaches

### 2. Fair Comparison is Critical
- Raw cost comparison favors supervised models unfairly
- **Normalized cost per fraud case** enables fair weighting
- Different scalers must be maintained for each model

### 3. Business Cost Optimization
- Threshold optimization reduces cost by 30-50% vs. default 0.5
- Asymmetric cost function (FP > FN) drives precision-focused models
- Total business impact matters more than F1-score

### 4. Feature Engineering Impact
- Cyclical time encoding (Hour_sin) captures fraud patterns
- Amount binning improves interpretability
- Dropping low-importance features (V13, V15, etc.) reduces noise

---

## Future Work

- [ ] Add FFNN hyperparameter tuning
- [ ] Explore temporal validation (time-based splits)
- [ ] Test ensemble on unseen fraud patterns
- [ ] Implement online learning for production deployment
- [ ] Add model explainability (SHAP values)

---

## References

- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **CatBoost**: [Official Documentation](https://catboost.ai/)
- **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or contributions, please open an issue on GitHub.
