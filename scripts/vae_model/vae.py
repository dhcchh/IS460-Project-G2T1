import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import time

class VAE(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=128, latent_dim=8):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.decoder_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.decoder_fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        h2 = F.relu(self.encoder_fc2(h1))
        return self.mu(h2), self.logvar(h2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        return self.decoder_fc3(h2)  # Remove sigmoid for continuous normalized data
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss - use MSE for continuous normalized data
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


class FraudVAEExperiment:
    """
    A complete experiment runner for VAE-based fraud detection.

    Usage:
        experiment = FraudVAEExperiment()
        experiment.load_data('../data/processed/creditcard_fe.csv')
        experiment.train(epochs=50)
        experiment.evaluate()
    """

    def __init__(self, input_dim=37, hidden_dim=128, latent_dim=8, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.model = VAE(input_dim, hidden_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_losses = []
        self.val_losses = []

    def load_data(self, data_path, test_size=0.2, val_size=0.2):
        """Load and preprocess the credit card fraud dataset."""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Separate features and target
        X = df.drop('Class', axis=1).values
        y = df['Class'].values

        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {y.mean():.4f} ({y.sum()} frauds out of {len(y)} transactions)")

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )

        # Use only normal transactions for training (VAE learns normal patterns)
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]

        print(f"Training on {len(X_train_normal)} normal transactions")
        print(f"Validation: {len(X_val)} samples ({y_val.sum()} frauds)")
        print(f"Test: {len(X_test)} samples ({y_test.sum()} frauds)")

        # Fit scaler on normal training data only
        self.scaler.fit(X_train_normal)

        # Transform all datasets
        X_train_normal = self.scaler.transform(X_train_normal)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train_normal).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = y_val
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = y_test

        print("Data loading complete!")

    def train(self, epochs=50, batch_size=256):
        """Train the VAE model."""
        print(f"\nStarting training for {epochs} epochs")
        print(f"Device: {self.device}")

        dataset = torch.utils.data.TensorDataset(self.X_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation loss
            val_loss = self._compute_validation_loss()
            self.train_losses.append(epoch_loss / len(self.X_train))
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}/{epochs} | Train Loss: {self.train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}')

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            recon_batch, mu, logvar = self.model(self.X_val)
            val_loss = vae_loss(recon_batch, self.X_val, mu, logvar).item() / len(self.X_val)
        self.model.train()
        return val_loss

    def compute_reconstruction_error(self, X):
        """Compute reconstruction error for anomaly detection."""
        self.model.eval()
        with torch.no_grad():
            recon_x, _, _ = self.model(X)
            # Mean squared error per sample
            errors = torch.mean((X - recon_x) ** 2, dim=1)
        return errors.cpu().numpy()

    def evaluate(self, threshold_percentile=95):
        """Evaluate the model on test data."""
        print(f"\n=== EVALUATION RESULTS ===")

        # Compute reconstruction errors
        val_errors = self.compute_reconstruction_error(self.X_val)
        test_errors = self.compute_reconstruction_error(self.X_test)

        # Set threshold based on validation normal transactions
        normal_val_errors = val_errors[self.y_val == 0]
        threshold = np.percentile(normal_val_errors, threshold_percentile)
        print(f"Anomaly threshold (validation {threshold_percentile}th percentile): {threshold:.6f}")

        # Make predictions
        test_predictions = (test_errors > threshold).astype(int)

        # Compute metrics
        roc_auc = roc_auc_score(self.y_test, test_errors)
        precision, recall, _ = precision_recall_curve(self.y_test, test_errors)
        pr_auc = auc(recall, precision)

        print(f"\nMetrics:")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"\nClassification Report (threshold = {threshold:.6f}):")
        print(classification_report(self.y_test, test_predictions,
                                  target_names=['Normal', 'Fraud']))

        # Plot results
        self._plot_results(val_errors, test_errors, threshold)

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'threshold': threshold,
            'test_errors': test_errors,
            'test_predictions': test_predictions
        }

    def _plot_results(self, val_errors, test_errors, threshold):
        """Plot training curves and error distributions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Training curves
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True)

        # Validation error distribution
        normal_val_errors = val_errors[self.y_val == 0]
        fraud_val_errors = val_errors[self.y_val == 1]

        ax2.hist(normal_val_errors, bins=50, alpha=0.7, label='Normal', density=True)
        ax2.hist(fraud_val_errors, bins=50, alpha=0.7, label='Fraud', density=True)
        ax2.axvline(threshold, color='red', linestyle='--', label=f'Threshold')
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Validation Error Distribution')
        ax2.legend()
        ax2.grid(True)

        # Test error distribution
        normal_test_errors = test_errors[self.y_test == 0]
        fraud_test_errors = test_errors[self.y_test == 1]

        ax3.hist(normal_test_errors, bins=50, alpha=0.7, label='Normal', density=True)
        ax3.hist(fraud_test_errors, bins=50, alpha=0.7, label='Fraud', density=True)
        ax3.axvline(threshold, color='red', linestyle='--', label=f'Threshold')
        ax3.set_xlabel('Reconstruction Error')
        ax3.set_ylabel('Density')
        ax3.set_title('Test Error Distribution')
        ax3.legend()
        ax3.grid(True)

        # Error comparison
        ax4.scatter(normal_test_errors[:1000], [0]*min(1000, len(normal_test_errors)),
                   alpha=0.6, label='Normal', s=10)
        ax4.scatter(fraud_test_errors, [1]*len(fraud_test_errors),
                   alpha=0.6, label='Fraud', s=10)
        ax4.axvline(threshold, color='red', linestyle='--', label=f'Threshold')
        ax4.set_xlabel('Reconstruction Error')
        ax4.set_ylabel('Class')
        ax4.set_title('Test Errors by Class')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


def run_basic_experiment():
    """
    Run a basic VAE fraud detection experiment.

    Usage:
        python scripts/vae/vae.py
    """
    print("=== VAE Fraud Detection Experiment ===\n")

    # Initialize experiment
    experiment = FraudVAEExperiment(input_dim=37, hidden_dim=128, latent_dim=8)

    # Load data
    experiment.load_data('../../data/processed/creditcard_fe.csv')

    # Train model
    experiment.train(epochs=50, batch_size=256)

    # Evaluate
    results = experiment.evaluate(threshold_percentile=95)

    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"PR AUC: {results['pr_auc']:.4f}")
    print(f"Anomaly threshold: {results['threshold']:.6f}")

    return experiment, results


if __name__ == "__main__":
    experiment, results = run_basic_experiment()