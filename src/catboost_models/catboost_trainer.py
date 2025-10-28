"""
CatBoost trainer: data prep, train, thresholding, eval, visualize, save.
"""

import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation import FraudEvaluationMetrics

from .catboost_base import (
    CatBoostDataHandler, ThresholdOptimizer, compute_prediction_scores
)

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise ImportError("CatBoost is required. Add 'catboost' to requirements.txt and install.")


class CatBoostFraudTrainer:
    """Train CatBoost for fraud detection with cost-based thresholding."""
    def __init__(self, config):
        """Store config and set up data/threshold utilities."""
        self.config = config
        self.random_seed = config.get('random_seed', 42)

        # Initialize components
        self.data_handler = CatBoostDataHandler(
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
        self.val_cost = None
        self.all_thresholds = None
        self.all_costs = None
        self.train_losses = []

    def prepare_data(self):
        """Load data and scale features."""
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
        """Init CatBoostClassifier with common params."""
        print("\n[3/6] Building CatBoost model...")
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:pr',
            'learning_rate': self.config.get('learning_rate', 0.05),
            'depth': self.config.get('depth', 6),
            'l2_leaf_reg': self.config.get('l2_leaf_reg', 3.0),
            'iterations': self.config.get('iterations', 500),
            'random_seed': self.random_seed,
            'verbose': False
        }
        # Optionally handle class imbalance
        if 'class_weights' in self.config:
            params['class_weights'] = self.config['class_weights']
        elif 'scale_pos_weight' in self.config:
            params['scale_pos_weight'] = self.config['scale_pos_weight']

        self.model = CatBoostClassifier(**params)
        print(f"  Learning rate: {params['learning_rate']}")
        print(f"  Depth: {params['depth']}")
        print(f"  Iterations: {params['iterations']}")
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        """Fit model on train and track best iteration."""
        print(f"\nTraining CatBoost model...")

        # Create data pools
        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_val, y_val)

        # Train model
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False)

        # Get metrics from training
        best_iteration = self.model.get_best_iteration()
        best_score = self.model.get_best_score()
        print(f"Training completed!")
        print(f"  Best iteration: {best_iteration}")
        print(f"  Best validation score: {best_score:.4f}")

    def optimize_threshold(self, X_val, y_val):
        """Pick threshold on val to minimize cost."""
        print(f"\n[4/6] Finding optimal business threshold...")

        # Get prediction probabilities
        val_probabilities = compute_prediction_scores(self.model, X_val)

        print(f"Testing threshold values from 0.1 to 0.9...")
        self.optimal_threshold, self.val_cost, self.all_thresholds, self.all_costs = \
            self.threshold_optimizer.find_optimal(val_probabilities, y_val)

        print(f"Optimal threshold found: {self.optimal_threshold:.4f}")
        print(f"Minimum validation cost: ${self.val_cost:,.0f}")

        return val_probabilities

    def evaluate(self, X_test, y_test):
        """Evaluate on test using chosen threshold and compute metrics."""
        print(f"\n[5/6] Evaluating on test set...")

        # Get prediction probabilities
        test_probabilities = compute_prediction_scores(self.model, X_test)
        y_pred = (test_probabilities >= self.optimal_threshold).astype(int)

        # Use evaluation module
        evaluator = FraudEvaluationMetrics(
            cost_fp=self.config['C_FP'],
            cost_fn=self.config['C_FN']
        )
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_scores=test_probabilities)

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
            'probabilities': test_probabilities,
            'predictions': y_pred
        }

    def visualize(self, val_probabilities, y_val, test_results, y_test, save_path):
        """Make 2x2 figure: importance, val/test dists, and cost curve."""
        print(f"\n[6/6] Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Subplot 1: Feature Importance
        ax = axes[0, 0]
        feature_names = self.model.feature_names_ if hasattr(self.model, 'feature_names_') else None
        importances = self.model.get_feature_importance()
        
        if feature_names and len(feature_names) == len(importances):
            # Top 20 features
            top_indices = np.argsort(importances)[-20:][::-1]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            ax.barh(range(len(top_features)), top_importances, color='steelblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features, fontsize=8)
        else:
            # Numeric labels
            top_indices = np.argsort(importances)[-20:][::-1]
            top_importances = importances[top_indices]
            ax.barh(range(len(top_importances)), top_importances, color='steelblue')
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels([f'Feature {i}' for i in top_indices], fontsize=8)
        
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Subplot 2: Validation Probability Distribution
        ax = axes[0, 1]
        normal_probs_val = val_probabilities[y_val == 0]
        fraud_probs_val = val_probabilities[y_val == 1]

        ax.hist(normal_probs_val, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_probs_val):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_probs_val, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_probs_val):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.4f}')
        ax.set_xlabel('Fraud Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Validation Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 3: Test Probability Distribution
        ax = axes[1, 0]
        test_probabilities = test_results['probabilities']
        normal_probs_test = test_probabilities[y_test == 0]
        fraud_probs_test = test_probabilities[y_test == 1]

        ax.hist(normal_probs_test, bins=50, alpha=0.7,
                label=f'Normal (n={len(normal_probs_test):,})',
                color='blue', edgecolor='black')
        ax.hist(fraud_probs_test, bins=50, alpha=0.7,
                label=f'Fraud (n={len(fraud_probs_test):,})',
                color='red', edgecolor='black')
        ax.axvline(self.optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.optimal_threshold:.4f}')
        ax.set_xlabel('Fraud Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Test Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 4: Business Cost Optimization
        ax = axes[1, 1]

        # Plot cost curve across all tested thresholds
        ax.plot(self.all_thresholds, self.all_costs, 'b-', linewidth=2)

        # Mark the optimal threshold
        optimal_idx = np.argmin(self.all_costs)
        ax.plot(self.all_thresholds[optimal_idx], self.all_costs[optimal_idx],
                'ro', markersize=10, label=f'Optimal: ${self.all_costs[optimal_idx]:,.0f}')

        ax.set_xlabel('Probability Threshold', fontsize=12)
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
        """Save CatBoost model and a small metadata file."""
        import pickle
        save_dict = {
            'model_path': save_path,
            'optimal_threshold': self.optimal_threshold,
            'scaler': self.scaler,
            'config': {
                'learning_rate': self.config.get('learning_rate'),
                'depth': self.config.get('depth'),
                'l2_leaf_reg': self.config.get('l2_leaf_reg'),
                'iterations': self.config.get('iterations'),
                'C_FP': self.config['C_FP'],
                'C_FN': self.config['C_FN']
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

        # Save CatBoost model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)

        # Save metadata
        metadata_path = save_path.replace('.cbm', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Model saved to: {save_path}")
        print(f"Metadata saved to: {metadata_path}")

    def run(self):
        """Run end-to-end pipeline and return final results."""
        print("=" * 60)
        print("CATBOOST FRAUD DETECTION TRAINING")
        print("=" * 60)

        # Create output directories
        os.makedirs(os.path.dirname(self.config.get('model_save_path', 'models/')), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.get('results_save_path', 'results/figures/')), exist_ok=True)

        # Prepare data
        data = self.prepare_data()

        # Build model
        self.build_model()

        # Train
        self.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

        # Optimize threshold
        val_probabilities = self.optimize_threshold(data['X_val'], data['y_val'])

        # Evaluate
        test_results = self.evaluate(data['X_test'], data['y_test'])

        # Visualize
        self.visualize(
            val_probabilities, data['y_val'],
            test_results, data['y_test'],
            self.config.get('results_save_path', 'results/figures/catboost_baseline_training.png')
        )

        # Save model
        self.save_model(
            test_results,
            self.config.get('model_save_path', 'models/catboost_baseline.cbm')
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)

        return test_results


