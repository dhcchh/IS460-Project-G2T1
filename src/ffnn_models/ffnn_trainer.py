"""
VAE Trainer Class

Handles single VAE training workflow with evaluation and saving.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation import FraudEvaluationMetrics

from .vae_base import (
    VAE, vae_loss, FraudDataHandler,
    compute_reconstruction_errors, ThresholdOptimizer
)


class VAEFraudTrainer:
    """
    Trains a single VAE model for fraud detection.

    Responsibilities:
    - Train VAE on normal transactions
    - Find optimal threshold on validation set
    - Evaluate on test set
    - Visualize results
    - Save model artifacts
    """

    def __init__(self, config):
        """
        Initialize trainer with configuration.

        Args:
            config: Dictionary with training parameters
                - data_path: Path to CSV file
                - drop_features: Features to drop (list or 'logreg_baseline')
                - hidden_dim: Hidden layer dimension
                - latent_dim: Latent dimension
                - beta: KL divergence weight
                - learning_rate: Optimizer learning rate
                - epochs: Number of training epochs
                - batch_size: Training batch size
                - C_FP: Cost per false positive
                - C_FN: Cost per false negative
                - device: 'cuda' or 'cpu'
                - random_seed: Random seed
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = config.get('random_seed', 42)

        # Set random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Initialize components
        self.data_handler = FraudDataHandler(
            config['data_path'],
            random_seed=self.random_seed,
            drop_features=config.get('drop_features', None)
        )
        self.threshold_optimizer = ThresholdOptimizer(
            C_FP=config.get('C_FP', 550),
            C_FN=config.get('C_FN', 110)
        )

        # Model and training state
        self.model = None
        self.input_dim = None
        self.scaler = None
        self.optimal_threshold = None
        self.train_losses = []
        self.val_losses = []

    def prepare_data(self):
        """Load and preprocess data."""
        print("\n[1/6] Loading and preparing data...")
        data = self.data_handler.load_and_split()
        self.input_dim = data['input_dim']

        print("\n[2/6] Scaling features...")
        X_train_scaled, X_val_scaled, X_test_scaled = self.data_handler.preprocess(
            data['X_train'], data['X_val'], data['X_test']
        )
        self.scaler = self.data_handler.scaler
        print("Features scaled using StandardScaler")

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test']
        }

    def build_model(self):
        """Initialize VAE model."""
        self.model = VAE(
            self.input_dim,
            self.config['hidden_dim'],
            self.config['latent_dim']
        ).to(self.device)
        return self.model

    def train(self, X_train, X_val, y_val):
        """
        Train VAE on normal transactions.

        Args:
            X_train: Training features (scaled)
            X_val: Validation features (scaled)
            y_val: Validation labels
        """
        print(f"\n[3/6] Training VAE on {self.device.upper()}...")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # Get only normal validation samples for validation loss
        normal_val_mask = (y_val == 0)
        X_val_normal = X_val_tensor[normal_val_mask]

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0

            # Create random batches
            num_samples = len(X_train_tensor)
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, self.config['batch_size']):
                batch_indices = indices[i:min(i + self.config['batch_size'], num_samples)]
                batch = X_train_tensor[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                loss = vae_loss(recon_batch, batch, mu, logvar, self.config['beta'])

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Average training loss
            avg_train_loss = train_loss / num_samples
            self.train_losses.append(avg_train_loss)

            # Validation phase (on normal validation data only)
            self.model.eval()
            with torch.no_grad():
                recon_val, mu_val, logvar_val = self.model(X_val_normal)
                val_loss = vae_loss(
                    recon_val, X_val_normal, mu_val, logvar_val, self.config['beta']
                )
                avg_val_loss = val_loss.item() / len(X_val_normal)
                self.val_losses.append(avg_val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs']} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        print(f"\nTraining completed!")
        print(f"  Final training loss: {self.train_losses[-1]:.4f}")
        print(f"  Final validation loss: {self.val_losses[-1]:.4f}")

    def optimize_threshold(self, X_val, y_val):
        """
        Find optimal threshold on validation set.

        Args:
            X_val: Validation features (scaled)
            y_val: Validation labels
        """
        print(f"\n[4/6] Finding optimal business threshold...")

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        val_errors = compute_reconstruction_errors(self.model, X_val_tensor)

        print(f"Testing threshold values from 50th to 99.9th percentile...")
        self.optimal_threshold, self.val_cost, self.all_thresholds, self.all_costs = \
            self.threshold_optimizer.find_optimal(val_errors, y_val)

        print(f"Optimal threshold found: {self.optimal_threshold:.6f}")
        print(f"Minimum validation cost: ${self.val_cost:,.0f}")

        return val_errors

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set using optimal threshold.

        Args:
            X_test: Test features (scaled)
            y_test: Test labels

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n[5/6] Evaluating on test set...")

        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        test_errors = compute_reconstruction_errors(self.model, X_test_tensor)
        y_pred = (test_errors > self.optimal_threshold).astype(int)

        # Use evaluation module
        evaluator = FraudEvaluationMetrics(
            cost_fp=self.config['C_FP'],
            cost_fn=self.config['C_FN']
        )
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_scores=test_errors)

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Baseline cost
        baseline_cost = y_test.sum() * self.config['C_FN']
        savings = baseline_cost - metrics['total_cost']

        # Print results
        print("\n" + "=" * 60)
        print("=== TEST SET EVALUATION ===")
        print("=" * 60)
        print("\nBusiness Cost Analysis:")
        print(f"  False Positives: {fp:,} × ${self.config['C_FP']} = "
              f"${int(metrics['false_positives']) * self.config['C_FP']:,}")
        print(f"  False Negatives: {fn:,} × ${self.config['C_FN']} = "
              f"${int(metrics['false_negatives']) * self.config['C_FN']:,}")
        print(f"  Total Cost: ${int(metrics['total_cost']):,}")
        print(f"\nBaseline Cost (no detection): ${baseline_cost:,}")
        print(f"Net Savings: ${int(savings):,}", end="")
        if savings < 0:
            print(" (WORSE than baseline)")
        else:
            print(" (BETTER than baseline)")

        print("\nConfusion Matrix:")
        print(f"  True Negatives: {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives: {tp:,}")

        print("\nClassification Metrics:")
        print(f"  Precision: {metrics['precision'] * 100:.1f}%")
        print(f"  Recall: {metrics['recall'] * 100:.1f}%")
        pr_auc_val = metrics['pr_auc'] if metrics['pr_auc'] is not None else 0.0
        print(f"  PR-AUC: {pr_auc_val:.4f}")

        return {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'total_cost': metrics['total_cost'],
            'baseline_cost': baseline_cost,
            'savings': savings,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'pr_auc': pr_auc_val,
            'errors': test_errors,
            'predictions': y_pred
        }

    def visualize(self, val_errors, y_val, test_results, y_test, save_path):
        """
        Create training and evaluation visualizations.

        Generates a 2x2 grid of plots showing:
        1. Training Curves (top-left): Training and validation loss over epochs
        2. Validation Error Distribution (top-right): Histogram of reconstruction
           errors for normal vs fraud transactions on validation set
        3. Test Error Distribution (bottom-left): Same as validation but on test set
        4. Business Cost Optimization (bottom-right): Shows how total business cost
           varies with different reconstruction error thresholds, highlighting the
           optimal threshold that minimizes cost

        The business cost plot helps visualize the trade-off between false positives
        (costly investigations) and false negatives (missed fraud). The x-axis shows
        different reconstruction error threshold values, and the y-axis shows the
        resulting total cost calculated as: Cost = (FP × $550) + (FN × $110)

        Args:
            val_errors: Validation reconstruction errors (numpy array)
            y_val: Validation labels (0=normal, 1=fraud)
            test_results: Test evaluation results dictionary containing:
                - 'errors': Test reconstruction errors
                - Other metrics (precision, recall, etc.)
            y_test: Test labels (0=normal, 1=fraud)
            save_path: Path to save the visualization figure (PNG)
        """
        print(f"\n[6/6] Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Subplot 1: Training and Validation Loss
        ax = axes[0, 0]
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'orange', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('VAE Training Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 2: Validation Error Distribution
        ax = axes[0, 1]
        normal_errors_val = val_errors[y_val == 0]
        fraud_errors_val = val_errors[y_val == 1]

        ax.hist(normal_errors_val, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_errors_val):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_errors_val, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_errors_val):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.4f}')
        ax.set_xlabel('Reconstruction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Validation Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 3: Test Error Distribution
        ax = axes[1, 0]
        test_errors = test_results['errors']
        normal_errors_test = test_errors[y_test == 0]
        fraud_errors_test = test_errors[y_test == 1]

        ax.hist(normal_errors_test, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_errors_test):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_errors_test, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_errors_test):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.4f}')
        ax.set_xlabel('Reconstruction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Test Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 4: Business Cost Optimization
        # This plot shows how total business cost varies with different threshold values.
        # X-axis: Reconstruction error threshold (higher = more strict fraud detection)
        # Y-axis: Total cost = (False Positives × $550) + (False Negatives × $110)
        # Goal: Find threshold that minimizes total business cost
        ax = axes[1, 1]

        # Plot cost curve across all tested thresholds
        ax.plot(self.all_thresholds, self.all_costs, 'b-', linewidth=2)

        # Mark the optimal threshold (minimum cost point) with red dot
        optimal_idx = np.argmin(self.all_costs)
        ax.plot(self.all_thresholds[optimal_idx], self.all_costs[optimal_idx],
                'ro', markersize=10, label=f'Optimal: ${self.all_costs[optimal_idx]:,.0f}')

        # Label axes clearly
        ax.set_xlabel('Reconstruction Error Threshold', fontsize=12)
        ax.set_ylabel('Total Business Cost ($)', fontsize=12)
        ax.set_title('Business Cost vs Detection Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        ax.annotate(f'Optimal Threshold\n{self.optimal_threshold:.4f}',
                    xy=(self.all_thresholds[optimal_idx], self.all_costs[optimal_idx]),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def save_model(self, test_results, save_path):
        """
        Save trained model and artifacts.

        Args:
            test_results: Test evaluation results
            save_path: Path to save model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimal_threshold': self.optimal_threshold,
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.config['hidden_dim'],
                'latent_dim': self.config['latent_dim'],
                'beta': self.config['beta'],
                'C_FP': self.config['C_FP'],
                'C_FN': self.config['C_FN']
            },
            'metrics': {
                'val_cost': self.val_cost,
                'test_cost': test_results['total_cost'],
                'test_precision': test_results['precision'],
                'test_recall': test_results['recall'],
                'pr_auc': test_results['pr_auc'],
                # Confusion matrix values
                'tn': test_results['tn'],
                'fp': test_results['fp'],
                'fn': test_results['fn'],
                'tp': test_results['tp']
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")

    def run(self):
        """
        Execute complete training pipeline.

        Returns:
            Dictionary with final results
        """
        print("=" * 60)
        print("VAE FRAUD DETECTION TRAINING")
        print("=" * 60)

        # Create output directories
        os.makedirs(os.path.dirname(self.config.get('model_save_path', 'models/')), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.get('results_save_path', 'results/figures/')), exist_ok=True)

        # Prepare data
        data = self.prepare_data()

        # Build model
        self.build_model()

        # Train
        self.train(data['X_train'], data['X_val'], data['y_val'])

        # Optimize threshold
        val_errors = self.optimize_threshold(data['X_val'], data['y_val'])

        # Evaluate
        test_results = self.evaluate(data['X_test'], data['y_test'])

        # Visualize
        self.visualize(
            val_errors, data['y_val'],
            test_results, data['y_test'],
            self.config.get('results_save_path', 'results/figures/vae_training_results.png')
        )

        # Save model
        self.save_model(
            test_results,
            self.config.get('model_save_path', 'models/vae_fraud_model.pth')
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)

        return test_results
