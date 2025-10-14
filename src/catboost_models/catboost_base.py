"""
CatBoost Base Utilities

Data handling and shared helpers for CatBoost fraud detection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CatBoostDataHandler:
    """
    Handles data loading, splitting, and preprocessing for fraud detection.
    Mirrors `FraudDataHandler` in `vae_models.vae_base` for consistency.
    """

    def __init__(self, data_path, random_seed=42, drop_features=None):
        self.data_path = data_path
        self.random_seed = random_seed
        self.drop_features = drop_features
        self.scaler = None
        self.input_dim = None

        if drop_features == 'logreg_baseline':
            self.drop_features = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',
                                  'Time', 'Amount', 'Hour', 'Hour_sin', 'log_Amount']

    def load_and_split(self):
        """
        Load data and perform stratified split for classification.
        Returns train/val/test with labels and metadata.
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        amounts = df['Amount'].values if 'Amount' in df.columns else np.zeros(len(df))
        y = df['Class'].values

        columns_to_drop = ['Class']
        if self.drop_features is not None and isinstance(self.drop_features, list):
            columns_to_drop.extend(self.drop_features)

        X = df.drop(columns=columns_to_drop).values
        self.input_dim = X.shape[1]

        print(f"Dataset loaded: {len(df)} rows, features: {self.input_dim}")
        print(f"  Normal: {(y == 0).sum()} | Fraud: {(y == 1).sum()}")

        # Stratified split: 60/20/20 overall, preserving class distribution
        X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
            X, y, np.arange(len(y)), test_size=0.4, random_state=self.random_seed, stratify=y
        )
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X_temp, y_temp, idx_temp, test_size=0.5, random_state=self.random_seed, stratify=y_temp
        )

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
        """
        Scale features using StandardScaler to match other baselines.
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled


