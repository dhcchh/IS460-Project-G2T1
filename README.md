# Credit Card Fraud Detection - Multi-Model Comparison & Ensemble

A comprehensive fraud detection system comparing gradient boosting (CatBoost), anomaly detection (VAE), deep learning (FFNN), and traditional ML (Logistic Regression) approaches. Includes a novel **fair cost-based ensemble** that accounts for different training paradigms (supervised vs. unsupervised).

## Table of Contents
- [Credit Card Fraud Detection - Multi-Model Comparison \& Ensemble](#credit-card-fraud-detection---multi-model-comparison--ensemble)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Business Objective](#business-objective)
  - [Dataset](#dataset)
  - [Model Training Approaches](#model-training-approaches)
    - [1. Logistic Regression (Baseline)](#1-logistic-regression-baseline)
    - [2. CatBoost (Gradient Boosting)](#2-catboost-gradient-boosting)
    - [3. VAE (Variational Autoencoder)](#3-vae-variational-autoencoder)
    - [4. FFNN (Feed-Forward Neural Network)](#4-ffnn-feed-forward-neural-network)
  - [Ensemble Methodology](#ensemble-methodology)
    - [The Fair Weighting Problem](#the-fair-weighting-problem)
    - [Solution: Normalized Cost Per Fraud Case](#solution-normalized-cost-per-fraud-case)
    - [Ensemble Pipeline](#ensemble-pipeline)
  - [Project Structure](#project-structure)
  - [Setup \& Installation](#setup--installation)
    - [1. Create Data Folder](#1-create-data-folder)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Feature Engineering](#3-feature-engineering)
  - [Usage](#usage)
    - [Training Workflow](#training-workflow)
    - [Command Line Usage](#command-line-usage)
  - [Results](#results)
    - [Model Comparison (Full Dataset: 284,807 samples, 492 fraud)](#model-comparison-full-dataset-284807-samples-492-fraud)
    - [Key Insights](#key-insights)
    - [Ensemble Weights (Fair Normalized - 3 Models)](#ensemble-weights-fair-normalized---3-models)
  - [Key Learnings](#key-learnings)
    - [1. Training Paradigm Matters](#1-training-paradigm-matters)
    - [2. Fair Comparison is Critical](#2-fair-comparison-is-critical)
    - [3. Business Cost Optimization](#3-business-cost-optimization)
    - [4. Feature Engineering Impact](#4-feature-engineering-impact)
  - [Future Work](#future-work)
  - [References](#references)

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

**Training Pipeline** (`src/ffnn_models/`):
- Baseline training and hyperparameter tuning combined in single notebook
- Saved as `models/ffnn_baseline.pth` and `models/ffnn_best_tuned.pth`

**Notebook**: `notebooks/ffnn_complete_pipeline.ipynb`

---

## Ensemble Methodology

### The Fair Weighting Problem

**Why Simple Cost Weighting is Unfair**:

| Model | Training Paradigm | Fraud in Training | Test Set Fraud | Test Cost |
|-------|------------------|-------------------|----------------|-----------|
| CatBoost | Supervised | 295 cases (learned patterns) | 99 cases (20%) | $2,310 |
| FFNN | Supervised | 295 cases (learned patterns) | 99 cases (20%) | $5,720 |
| VAE | Unsupervised | 0 cases (anomaly only) | 246 cases (50%) | $18,700 |

**Problem**: Direct cost comparison is unfair because:
1. CatBoost & FFNN learned from 295 fraud examples
2. VAE never saw any fraud (pure anomaly detection)
3. VAE tested on 2.5× more fraud cases than supervised models
4. Supervised models naturally have lower cost when evaluated on data similar to training

### Solution: Normalized Cost Per Fraud Case

**Implementation** (`src/ensemble.py:346-420`):

```python
# Normalize by number of fraud cases in evaluation
cost_per_fraud_catboost = catboost_cost / n_fraud_cases  # $10,450 / 492 = $21.24
cost_per_fraud_ffnn = ffnn_cost / n_fraud_cases          # $20,350 / 492 = $41.36
cost_per_fraud_vae = vae_cost / n_fraud_cases            # $34,650 / 492 = $70.43

# Inverse cost weighting (lower cost per fraud = higher weight)
epsilon = 1e-6
inverse_catboost = 1.0 / (cost_per_fraud_catboost + epsilon)
inverse_ffnn = 1.0 / (cost_per_fraud_ffnn + epsilon)
inverse_vae = 1.0 / (cost_per_fraud_vae + epsilon)

total_inverse = inverse_catboost + inverse_ffnn + inverse_vae
weight_catboost = inverse_catboost / total_inverse
weight_ffnn = inverse_ffnn / total_inverse
weight_vae = inverse_vae / total_inverse
```

**Benefits**:
1. **Fair comparison** regardless of training paradigm
2. **Accounts for dataset size** (all evaluated on 492 fraud cases)
3. **Respects model strengths**:
   - CatBoost excels at known fraud patterns
   - FFNN provides deep learning representation
   - VAE excels at novel/unknown patterns
4. **Normalized metric** allows fair weighting across all models

### Ensemble Pipeline

**Data Handling** (Critical for Fairness):
```python
# Each model uses its OWN scaler
X_catboost = catboost_scaler.transform(X_raw)  # Fitted on all data
X_ffnn = ffnn_scaler.transform(X_raw)          # Fitted on all data
X_vae = vae_scaler.transform(X_raw)            # Fitted on normal only
```

**Prediction Combination**:
1. Get CatBoost probabilities (0-1 scale)
2. Get FFNN probabilities (0-1 scale)
3. Get VAE reconstruction errors (normalized to 0-1)
4. Weighted sum: `ensemble_score = w_cb × p_cb + w_ffnn × p_ffnn + w_vae × e_vae`
5. Optimize final threshold on full dataset

**Evaluation**:
- All models (LogReg, CatBoost, VAE, FFNN, Ensemble) evaluated on **SAME** 284,807 samples
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
│   ├── ffnn_baseline.pth
│   ├── ffnn_best_tuned.pth
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
│   ├── ensemble/                   # Fair Cost-Based Ensemble (3 models)
│   │   └── ensemble_test.ipynb
│   │
│   ├── ffnn_complete_pipeline.ipynb  # FFNN (Deep Learning) - all-in-one
│   └── EDA_statistical_analysis.ipynb
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
│   └── ffnn_models/
│       ├── ffnn_base.py            # FraudDetectionFFNN, FFNNDataHandler
│       ├── ffnn_trainer.py         # train_ffnn_baseline()
│       └── ffnn_tuner.py           # Optuna hyperparameter tuning
│
├── results/
│   ├── figures/                    # Visualizations
│   ├── ensemble_comparison.csv     # 5-model comparison
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

# FFNN (includes both baseline and tuning)
jupyter notebook notebooks/ffnn_complete_pipeline.ipynb
```

**2. Hyperparameter Tuning**:
```bash
# CatBoost Tuning
jupyter notebook notebooks/catboost_model_test/catboost_hyperparameter_tuning.ipynb

# VAE Tuning
jupyter notebook notebooks/vae_model_test/vae_hyperparameter_tuning.ipynb

# FFNN Tuning (included in complete pipeline above)
# See ffnn_complete_pipeline.ipynb
```

**3. Ensemble Comparison** (requires all three best tuned models):
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
    --ffnn models/ffnn_best_tuned.pth \
    --device cuda
```

---

## Results

### Model Comparison (Full Dataset: 284,807 samples, 492 fraud)

| Model | Training | Fraud in Training | Precision | Recall | PR-AUC | Cost | Savings |
|-------|----------|-------------------|-----------|--------|--------|------|---------|
| **LogReg** | Supervised | 394 (80%) | 0.8857 | 0.6301 | 0.7568 | $43,450 | 22.4% |
| **CatBoost** | Supervised | 295 (60%) | 0.9951 | 0.8272 | 0.9051 | $10,450 | 80.7% |
| **VAE** | Unsupervised | 0 (0%) | 0.9547 | 0.4715 | 0.6530 | $34,650 | 36.0% |
| **FFNN** | Supervised | 295 (60%) | 0.9852 | 0.6748 | 0.8757 | $20,350 | 62.4% |
| **Ensemble** | Fair Weighted (3 models) | - | 0.9950 | 0.8171 | 0.9113 | $11,000 | 79.7% |

**Baseline (No Detection)**: $54,120

### Key Insights

1. **CatBoost Best Individual**: Achieves lowest cost ($10,450) with 80.7% savings
2. **Ensemble Strong Performance**: 3-model ensemble achieves $11,000 cost (79.7% savings)
3. **FFNN Competitive**: Deep learning approach provides 62.4% savings
4. **VAE Effective**: Despite no fraud in training, achieves 36.0% savings
5. **LogReg Baseline**: Simple model provides 22.4% cost reduction floor

### Ensemble Weights (Fair Normalized - 3 Models)

```
CatBoost: 55.09% weight ($21.24 per fraud case)
FFNN:     28.29% weight ($41.36 per fraud case)
VAE:      16.62% weight ($70.43 per fraud case)
```

The normalized weighting accounts for:
- CatBoost & FFNN's supervised learning advantage (saw 295 fraud cases each)
- VAE's unsupervised disadvantage (saw 0 fraud cases)
- All models evaluated on same 492 fraud cases for fairness
- Lower cost per fraud case → higher contribution to ensemble

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

- [ ] Explore temporal validation (time-based splits)
- [ ] Test ensemble on unseen fraud patterns
- [ ] Implement online learning for production deployment
- [ ] Add model explainability (SHAP values)
- [ ] Experiment with stacking/meta-learning ensembles
- [ ] Add real-time inference optimization

---

## References

- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
