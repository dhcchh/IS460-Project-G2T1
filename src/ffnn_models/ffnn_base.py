"""
FFNN Base Classes and Components

This module contains core FFNN components used by both training and tuning.
Mirrors the structure of CatBoost/VAE base modules for consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


# ============= FFNN MODEL =============
class FraudDetectionFFNN(nn.Module):
    """
    Feed-forward neural network for fraud detection.
    
    Architecture:
        input_dim -> fc1 (hidden_dim) -> fc2 (hidden_dim//2) 
        -> fc3 (hidden_dim//4) -> output (1)
        
    With ReLU activations, dropout regularization, and sigmoid output.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(FraudDetectionFFNN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# ============= DATASET CLASS =============
class FraudDataset(Dataset):
    """
    PyTorch Dataset wrapper for fraud detection data.
    Handles conversion of numpy/pandas data to PyTorch tensors.
    """
    
    def __init__(self, X, y):
        """
        Initialize dataset.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            y: Labels (numpy array or pandas Series)
        """
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============= DATA HANDLER =============
class FFNNDataHandler:
    """
    Handles data loading, splitting, and preprocessing for FFNN fraud detection.
    Mirrors CatBoostDataHandler and FraudDataHandler for consistency.
    """
    
    def __init__(self, data_path, random_seed=42, drop_features=None, test_size=0.2, val_size=0.2):
        """
        Initialize data handler.
        
        Args:
            data_path: Path to CSV file
            random_seed: Random seed for reproducibility
            drop_features: List of feature columns to drop (besides 'Class').
                          Use 'logreg_baseline' to match logistic regression features.
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.drop_features = drop_features
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = None
        self.input_dim = None
        self.class_weights = None
        self.feature_names = None
        
        # Preset for logistic regression baseline features
        if drop_features == 'logreg_baseline':
            self.drop_features = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',
                                  'Time', 'Amount', 'Hour', 'Hour_sin', 'log_Amount']
    
    def load_and_split(self):
        """
        Load data and perform stratified split for binary classification.
        Returns train/val/test splits with labels, amounts, and class weights.
        
        Returns:
            Dictionary with train/val/test data and metadata
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Extract amounts before dropping features
        amounts = df['Amount'].values if 'Amount' in df.columns else np.zeros(len(df))
        y = df['Class'].values
        
        # Drop Class column and specified features
        columns_to_drop = ['Class']
        if self.drop_features is not None and isinstance(self.drop_features, list):
            columns_to_drop.extend(self.drop_features)
        
        X = df.drop(columns=columns_to_drop).values
        self.feature_names = df.drop(columns=columns_to_drop).columns.tolist()
        self.input_dim = X.shape[1]
        
        print(f"Dataset loaded: {len(df)} rows, features: {self.input_dim}")
        print(f"  Normal: {(y == 0).sum()} | Fraud: {(y == 1).sum()}")
        print(f"  Fraud rate: {(y == 1).sum() / len(y):.2%}")
        
        # Calculate balanced class weights for imbalanced data
        self.class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y), y=y
        )
        self.class_weights = {i: w for i, w in enumerate(self.class_weights)}
        print(f"  Class weights: {self.class_weights}")
        
        # Stratified split: train/val/test
        test_size_fraction = self.test_size
        val_size_fraction = self.val_size / (1 - self.test_size)
        
        # First split: train+val vs test
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)),
            test_size=test_size_fraction,
            random_state=self.random_seed,
            stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_train, y_train, idx_train,
            test_size=val_size_fraction,
            random_state=self.random_seed,
            stratify=y_train
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples ({(y_train == 1).sum()} fraud)")
        print(f"  Validation: {len(X_val)} samples ({(y_val == 1).sum()} fraud)")
        print(f"  Test: {len(X_test)} samples ({(y_test == 1).sum()} fraud)")
        
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
            'input_dim': self.input_dim,
            'class_weights': self.class_weights
        }
    
    def preprocess(self, X_train, X_val, X_test):
        """
        Normalize features using StandardScaler.
        Fit on training data and apply to validation/test sets.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data normalized:")
        print(f"  Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        print(f"  Val mean: {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_class_weights_dict(self):
        """Returns class weights as dictionary for use in model training."""
        return self.class_weights
    
    def get_feature_names(self):
        """Returns list of feature names after dropping specified features."""
        return self.feature_names


# ============= PREDICTION SCORES =============
def compute_prediction_scores(model, X_tensor, device='cpu'):
    """
    Compute prediction probabilities for each sample.
    
    Args:
        model: Trained FFNN model
        X_tensor: Input tensor (already on correct device or will be moved)
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        Numpy array of fraud probabilities (one per sample)
    """
    model.eval()
    with torch.no_grad():
        if not isinstance(X_tensor, torch.Tensor):
            X_tensor = torch.FloatTensor(X_tensor)
        X_tensor = X_tensor.to(device)
        scores = model(X_tensor)
    return scores.cpu().numpy().ravel()


# ============= THRESHOLD OPTIMIZER =============
class ThresholdOptimizer:
    """
    Finds optimal threshold to minimize business cost.
    Mirrors the CatBoost and VAE threshold optimizers.
    """
    
    def __init__(self, C_FP=550, C_FN=110):
        """
        Initialize optimizer with cost parameters.
        
        Args:
            C_FP: Cost per false positive (flagging legitimate transaction as fraud)
            C_FN: Cost per false negative (missing actual fraud)
        """
        self.C_FP = C_FP
        self.C_FN = C_FN
    
    def find_optimal(self, scores, y_true, threshold_range=(0.1, 0.9), step=0.01):
        """
        Find threshold that minimizes business cost.
        Cost = (FP × C_FP) + (FN × C_FN)
        
        Args:
            scores: Prediction probabilities/scores
            y_true: True labels (0=normal, 1=fraud)
            threshold_range: (min, max) threshold range to search
            step: Step size for threshold search
        
        Returns:
            Tuple of (optimal_threshold, minimum_cost, all_thresholds, all_costs)
        """
        # Try thresholds in specified range
        thresholds = np.arange(threshold_range[0], threshold_range[1], step)
        costs = []
        
        for threshold in thresholds:
            # Predict: score >= threshold → fraud (1), else normal (0)
            y_pred = (scores >= threshold).astype(int)
            
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