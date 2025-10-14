"""
VAE Base Classes and Components

This module contains core VAE components used by both training and tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ============= VAE MODEL =============
class VAE(nn.Module):
    """
    Variational Autoencoder for anomaly detection.

    Architecture:
        Encoder: input_dim -> hidden_dim -> hidden_dim//2 -> latent_dim
        Decoder: latent_dim -> hidden_dim//2 -> hidden_dim -> input_dim
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encode input to mu and logvar."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from N(mu, sigma).
        z = mu + epsilon * sigma where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)  # No activation for normalized data

    def forward(self, x):
        """Complete forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ============= LOSS FUNCTION =============
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = MSE reconstruction + beta * KL divergence

    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Mean from encoder
        logvar: Log variance from encoder
        beta: Weight for KL term

    Returns:
        Total loss (scalar)
    """
    # Reconstruction loss (MSE)
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + beta * kl_loss


# ============= DATA HANDLER =============
class FraudDataHandler:
    """
    Handles data loading, splitting, and preprocessing for fraud detection.
    """
    def __init__(self, data_path, random_seed=42, drop_features=None):
        """
        Initialize data handler.

        Args:
            data_path: Path to CSV file
            random_seed: Random seed for reproducibility
            drop_features: List of feature columns to drop (besides 'Class').
                          Use 'logreg_baseline' to match logistic regression features.
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.drop_features = drop_features
        self.scaler = None
        self.input_dim = None

        # Preset for logistic regression baseline features
        # Model expects: V1-V12, V14, V16-V18, V21, V27-V28, Day, Time_bin, Amount_bin (22 features)
        if drop_features == 'logreg_baseline':
            self.drop_features = ['V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',
                                 'Time', 'Amount', 'Hour', 'Hour_sin', 'log_Amount']

    def load_and_split(self):
        """
        Load data and split properly for fraud detection.

        Training: Only normal transactions (60% of normal data)
        Validation: Normal + Fraud (20% normal, 50% fraud)
        Test: Normal + Fraud (20% normal, 50% fraud)

        Returns:
            Dictionary with all data splits and metadata
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Save amounts before dropping features (if needed)
        amounts = df['Amount'].values if 'Amount' in df.columns else np.zeros(len(df))

        # Separate features and labels
        y = df['Class'].values

        # Drop Class and optionally other features
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

        # Get indices for each class
        normal_idx = np.where(y == 0)[0]
        fraud_idx = np.where(y == 1)[0]

        # Split normal: 60/20/20
        normal_train, normal_temp = train_test_split(
            normal_idx, test_size=0.4, random_state=self.random_seed
        )
        normal_val, normal_test = train_test_split(
            normal_temp, test_size=0.5, random_state=self.random_seed
        )

        # Split fraud: 0/50/50 (no fraud in training!)
        fraud_val, fraud_test = train_test_split(
            fraud_idx, test_size=0.5, random_state=self.random_seed
        )

        # Combine indices
        train_idx = normal_train  # Only normal
        val_idx = np.concatenate([normal_val, fraud_val])
        test_idx = np.concatenate([normal_test, fraud_test])

        # Shuffle validation and test sets
        np.random.seed(self.random_seed)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        print("\nData split:")
        print(f"  Training: {len(train_idx)} normal transactions")
        print(f"  Validation: {len(val_idx)} transactions ({(y[val_idx] == 1).sum()} fraud)")
        print(f"  Test: {len(test_idx)} transactions ({(y[test_idx] == 1).sum()} fraud)")

        return {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'amounts_train': amounts[train_idx],
            'X_val': X[val_idx],
            'y_val': y[val_idx],
            'amounts_val': amounts[val_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx],
            'amounts_test': amounts[test_idx],
            'input_dim': self.input_dim
        }

    def preprocess(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler.
        Fit only on training data to prevent data leakage.

        Returns:
            Scaled arrays and fitted scaler
        """
        self.scaler = StandardScaler()

        # Fit only on training data
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Transform validation and test data
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled


# ============= ERROR COMPUTATION =============
def compute_reconstruction_errors(model, X_tensor):
    """
    Compute reconstruction error for each sample.
    Error = mean((x - reconstruction)^2) for each sample

    Args:
        model: Trained VAE
        X_tensor: Input tensor on correct device

    Returns:
        Numpy array of errors (one per sample)
    """
    model.eval()
    with torch.no_grad():
        recon_x, _, _ = model(X_tensor)
        # MSE per sample (mean over features, not samples)
        errors = torch.mean((X_tensor - recon_x) ** 2, dim=1)
    return errors.cpu().numpy()


# ============= THRESHOLD OPTIMIZER =============
class ThresholdOptimizer:
    """
    Finds optimal threshold to minimize business cost.
    """
    def __init__(self, C_FP=550, C_FN=110):
        """
        Initialize optimizer with cost parameters.

        Args:
            C_FP: Cost per false positive
            C_FN: Cost per false negative
        """
        self.C_FP = C_FP
        self.C_FN = C_FN

    def find_optimal(self, errors, y_true, percentile_range=(50, 99.9), step=0.1):
        """
        Find threshold that minimizes business cost.
        Cost = (FP × C_FP) + (FN × C_FN)

        Args:
            errors: Reconstruction errors
            y_true: True labels
            percentile_range: (min, max) percentile range to search (default: 50-99.9)
            step: Step size for percentile search

        Returns:
            optimal_threshold, minimum_cost, all_thresholds, all_costs

        Note: Expanded from (85, 99.5) to (50, 99.9) to reduce false positives
        and better optimize for the business cost function where FP cost > FN cost.
        """
        # Try thresholds from percentile range
        percentiles = np.arange(percentile_range[0], percentile_range[1], step)
        thresholds = np.percentile(errors, percentiles)
        costs = []

        for threshold in thresholds:
            # Predict: error > threshold → fraud (1), else normal (0)
            y_pred = (errors > threshold).astype(int)

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
