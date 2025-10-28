"""
CatBoost base utilities: data handling, threshold search, and score helpers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============= DATA HANDLER =============
class CatBoostDataHandler:
    """
    Load data, make stratified splits, and scale features.
    """
    def __init__(self, data_path, random_seed=42, drop_features=None):
        """Configure paths, split seed, and optional features to drop."""
        self.data_path = data_path
        self.random_seed = random_seed
        self.drop_features = drop_features
        self.scaler = None
        self.input_dim = None

        # Preset to match logistic regression feature set
        if drop_features == 'logreg_baseline':
            self.drop_features = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',
                                 'Time', 'Amount', 'Hour', 'Hour_sin', 'log_Amount']

    def load_and_split(self):
        """
        Stratified 60/20/20 split with basic dataset stats.
        Returns dict with X/y splits, amounts, and input_dim.
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Keep raw amounts for optional business-cost analysis
        amounts = df['Amount'].values if 'Amount' in df.columns else np.zeros(len(df))
        y = df['Class'].values

        # Drop Class column and optionally other features
        columns_to_drop = ['Class']
        if self.drop_features is not None and isinstance(self.drop_features, list):
            columns_to_drop.extend(self.drop_features)

        X = df.drop(columns=columns_to_drop).values
        self.input_dim = X.shape[1]

        print(f"Dataset loaded: {len(df)} transactions")
        print(f"  Normal: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"  Fraud: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
        print(f"  Features: {self.input_dim}")
        if self.drop_features:
            print(f"  Dropped features: {len(self.drop_features) if isinstance(self.drop_features, list) else 'logreg_baseline preset'}")

        # Stratified split: 60/20/20 overall
        # First: 60% train, 40% temp
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(y)), test_size=0.4, random_state=self.random_seed, stratify=y
        )
        # Second: 20% val, 20% test from temp
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp, test_size=0.5, random_state=self.random_seed, stratify=y_temp
        )

        print("\nData split:")
        print(f"  Training: {len(X_train)} transactions")
        print(f"  Validation: {len(X_val)} transactions ({(y_val == 1).sum()} fraud)")
        print(f"  Test: {len(X_test)} transactions ({(y_test == 1).sum()} fraud)")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'amounts_train': amounts[idx_train],
            'X_val': X_val,
            'y_val': y_val,
            'amounts_val': amounts[idx_val],
            'X_test': X_test,
            'y_test': y_test,
            'amounts_test': amounts[idx_test],
            'input_dim': self.input_dim
        }

    def preprocess(self, X_train, X_val, X_test):
        """Fit StandardScaler on train; transform val/test."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled


# ============= THRESHOLD OPTIMIZER =============
class ThresholdOptimizer:
    """Search a probability threshold that minimizes business cost."""
    def __init__(self, C_FP=550, C_FN=110):
        """Set FP/FN costs."""
        self.C_FP = C_FP
        self.C_FN = C_FN

    def find_optimal(self, probabilities, y_true, threshold_range=(0.1, 0.9), step=0.01):
        """Return (best_threshold, min_cost, thresholds, costs)."""
        # Try thresholds in specified range
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        costs = []

        for threshold in thresholds:
            # Predict: prob >= threshold → fraud (1), else normal (0)
            y_pred = (probabilities >= threshold).astype(int)

            # Count false positives and false negatives
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()

            # Calculate business cost
            cost = fp * self.C_FP + fn * self.C_FN
            costs.append(cost)

        # Find threshold with minimum cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        min_cost = costs[optimal_idx]

        return optimal_threshold, min_cost, thresholds, costs


# ============= PREDICTION SCORE COMPUTATION =============
def compute_prediction_scores(model, X):
    """Return model predict_proba[:, 1] as numpy array."""
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities