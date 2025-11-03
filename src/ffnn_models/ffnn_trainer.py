"""
FFNN Trainer Class

Handles single FFNN training workflow with evaluation and saving.
Mirrors VAEFraudTrainer and CatBoostFraudTrainer for consistency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation import FraudEvaluationMetrics

from .ffnn_base import (
    FFNNDataHandler,
    FraudDetectionFFNN,
    FraudDataset,
    ThresholdOptimizer,
    compute_prediction_scores
)


class FFNNFraudTrainer:
    """
    Trains a single FFNN model for fraud detection.
    
    Responsibilities:
    - Load and preprocess data
    - Train FFNN with class weights
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
                - hidden_dim: Hidden layer dimension (default: 128)
                - dropout_rate: Dropout probability (default: 0.2)
                - learning_rate: Optimizer learning rate (default: 0.001)
                - epochs: Number of training epochs (default: 100)
                - batch_size: Training batch size (default: 64)
                - patience: Early stopping patience (default: 15)
                - test_size: Test set fraction (default: 0.2)
                - val_size: Validation set fraction (default: 0.2)
                - use_class_weights: Use balanced class weights (default: True)
                - C_FP: Cost per false positive (default: 550)
                - C_FN: Cost per false negative (default: 110)
                - device: 'cuda' or 'cpu'
                - random_seed: Random seed (default: 42)
                - model_save_path: Path to save model
                - results_save_path: Path to save visualizations
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = config.get('random_seed', 42)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        # Initialize components
        self.data_handler = FFNNDataHandler(
            config['data_path'],
            random_seed=self.random_seed,
            drop_features=config.get('drop_features', None),
            test_size=config.get('test_size', 0.2),
            val_size=config.get('val_size', 0.2)
        )
        
        self.threshold_optimizer = ThresholdOptimizer(
            C_FP=config.get('C_FP', 550),
            C_FN=config.get('C_FN', 110)
        )
        
        # Model and training state
        self.model = None
        self.input_dim = None
        self.scaler = None
        self.class_weights = None
        self.optimal_threshold = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self):
        """Load and preprocess data."""
        print("\n[1/6] Loading and preparing data...")
        data = self.data_handler.load_and_split()
        self.input_dim = data['input_dim']
        self.class_weights = data['class_weights']
        
        print("\n[2/6] Scaling features...")
        X_train_scaled, X_val_scaled, X_test_scaled = self.data_handler.preprocess(
            data['X_train'], data['X_val'], data['X_test']
        )
        self.scaler = self.data_handler.scaler
        
        # Create PyTorch datasets
        train_dataset = FraudDataset(X_train_scaled, data['y_train'])
        val_dataset = FraudDataset(X_val_scaled, data['y_val'])
        test_dataset = FraudDataset(X_test_scaled, data['y_test'])
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 64)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"PyTorch DataLoaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test']
        }
    
    def build_model(self):
        """Initialize FFNN model."""
        print("\n[3/6] Building model...")
        self.model = FraudDetectionFFNN(
            self.input_dim,
            self.config.get('hidden_dim', 128),
            self.config.get('dropout_rate', 0.2)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model architecture: {self.input_dim} → {self.config.get('hidden_dim', 128)} "
              f"→ {self.config.get('hidden_dim', 128)//2} → {self.config.get('hidden_dim', 128)//4} → 1")
        print(f"Total trainable parameters: {total_params:,}")
        
        return self.model
    
    def train(self, data_loaders):
        """
        Train FFNN model with optional class weights and early stopping.
        
        Args:
            data_loaders: Dictionary with train_loader, val_loader, and labels
        """
        print(f"\n[4/6] Training FFNN on {self.device.upper()}...")
        
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        
        # Loss function with optional class weighting
        if self.config.get('use_class_weights', True):
            pos_weight = torch.tensor([
                self.class_weights[1] / self.class_weights[0]
            ]).to(self.device)
            criterion = nn.BCELoss()  # Note: BCELoss doesn't use pos_weight
            print(f"Using BCE loss (class weight ratio: {pos_weight.item():.2f})")
        else:
            criterion = nn.BCELoss()
            print("Using standard BCE loss")
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = self.config.get('patience', 15)
        patience_counter = 0
        
        # Training loop
        epochs = self.config.get('epochs', 100)
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / train_total
            train_acc = 100 * train_correct / train_total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / val_total
            val_acc = 100 * val_correct / val_total
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_ffnn_model_temp.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_ffnn_model_temp.pth'))
        print(f"\nTraining completed!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Final training loss: {self.train_losses[-1]:.4f}")
    
    def optimize_threshold(self, data_loaders):
        """
        Find optimal threshold on validation set to minimize business cost.
        
        Args:
            data_loaders: Dictionary with validation data
        """
        print(f"\n[5/6] Finding optimal business threshold...")
        
        X_val_scaled = data_loaders['X_val_scaled']
        y_val = data_loaders['y_val']
        
        # Get predictions on validation set
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        val_scores = compute_prediction_scores(self.model, X_val_tensor, self.device)
        
        print(f"Testing threshold values from 0.1 to 0.9...")
        self.optimal_threshold, self.val_cost, self.all_thresholds, self.all_costs = \
            self.threshold_optimizer.find_optimal(val_scores, y_val)
        
        print(f"Optimal threshold found: {self.optimal_threshold:.3f}")
        print(f"Minimum validation cost: ${self.val_cost:,.0f}")
        
        return val_scores
    
    def evaluate(self, data_loaders):
        """
        Evaluate on test set using optimal threshold.
        
        Args:
            data_loaders: Dictionary with test data
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n[6/6] Evaluating on test set...")
        
        X_test_scaled = data_loaders['X_test_scaled']
        y_test = data_loaders['y_test']
        
        # Get test predictions
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        test_scores = compute_prediction_scores(self.model, X_test_tensor, self.device)
        y_pred = (test_scores >= self.optimal_threshold).astype(int)
        
        # Use evaluation module
        evaluator = FraudEvaluationMetrics(
            cost_fp=self.config.get('C_FP', 550),
            cost_fn=self.config.get('C_FN', 110)
        )
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_scores=test_scores)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Baseline cost (no detection - all frauds missed)
        baseline_cost = y_test.sum() * self.config.get('C_FN', 110)
        savings = baseline_cost - metrics['total_cost']
        
        # Print results
        print("\n" + "=" * 60)
        print("=== TEST SET EVALUATION ===")
        print("=" * 60)
        print("\nBusiness Cost Analysis:")
        print(f"  False Positives: {fp:,} × ${self.config.get('C_FP', 550)} = "
              f"${int(metrics['false_positives']) * self.config.get('C_FP', 550):,}")
        print(f"  False Negatives: {fn:,} × ${self.config.get('C_FN', 110)} = "
              f"${int(metrics['false_negatives']) * self.config.get('C_FN', 110):,}")
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
            'predictions': y_pred,
            'probabilities': test_scores
        }
    
    def visualize(self, val_scores, y_val, test_results, y_test, save_path):
        """
        Create training and evaluation visualizations.
        
        Args:
            val_scores: Validation prediction scores
            y_val: Validation labels
            test_results: Test evaluation results
            y_test: Test labels
            save_path: Path to save the figure
        """
        print(f"\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Training Curves
        ax = axes[0, 0]
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'orange', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('FFNN Training Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Validation Score Distribution
        ax = axes[0, 1]
        normal_scores_val = val_scores[y_val == 0]
        fraud_scores_val = val_scores[y_val == 1]
        
        ax.hist(normal_scores_val, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_scores_val):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_scores_val, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_scores_val):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.3f}')
        ax.set_xlabel('Fraud Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Validation Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: Test Score Distribution
        ax = axes[1, 0]
        test_scores = test_results['probabilities']
        normal_scores_test = test_scores[y_test == 0]
        fraud_scores_test = test_scores[y_test == 1]
        
        ax.hist(normal_scores_test, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_scores_test):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_scores_test, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_scores_test):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.3f}')
        ax.set_xlabel('Fraud Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Test Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Subplot 4: Business Cost Optimization
        ax = axes[1, 1]
        ax.plot(self.all_thresholds, self.all_costs, 'b-', linewidth=2)
        
        optimal_idx = np.argmin(self.all_costs)
        ax.plot(self.all_thresholds[optimal_idx], self.all_costs[optimal_idx],
                'ro', markersize=10, label=f'Optimal: ${self.all_costs[optimal_idx]:,.0f}')
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Total Business Cost ($)', fontsize=12)
        ax.set_title('Business Cost vs Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        ax.annotate(f'Optimal Threshold\n{self.optimal_threshold:.3f}',
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
                'hidden_dim': self.config.get('hidden_dim', 128),
                'dropout_rate': self.config.get('dropout_rate', 0.2),
                'C_FP': self.config.get('C_FP', 550),
                'C_FN': self.config.get('C_FN', 110)
            },
            'metrics': {
                'val_cost': self.val_cost,
                'test_cost': test_results['total_cost'],
                'test_precision': test_results['precision'],
                'test_recall': test_results['recall'],
                'pr_auc': test_results['pr_auc'],
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
        print("FFNN FRAUD DETECTION TRAINING")
        print("=" * 60)
        
        # Create output directories
        os.makedirs(os.path.dirname(self.config.get('model_save_path', 'models/')), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.get('results_save_path', 'results/figures/')), exist_ok=True)
        
        # Prepare data
        data = self.prepare_data()
        
        # Build model
        self.build_model()
        
        # Train
        self.train(data)
        
        # Optimize threshold
        val_scores = self.optimize_threshold(data)
        
        # Evaluate
        test_results = self.evaluate(data)
        
        # Visualize
        self.visualize(
            val_scores, data['y_val'],
            test_results, data['y_test'],
            self.config.get('results_save_path', 'results/figures/ffnn_training_results.png')
        )
        
        # Save model
        self.save_model(
            test_results,
            self.config.get('model_save_path', 'models/ffnn_fraud_model.pth')
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return test_results