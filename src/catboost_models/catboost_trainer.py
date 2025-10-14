"""
CatBoost Trainer

End-to-end training, evaluation, visualization, and saving pipeline for CatBoost.
Mirrors the structure and responsibilities of `VAEFraudTrainer`.
"""

import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path to use shared evaluation utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation import FraudEvaluationMetrics

from .catboost_base import CatBoostDataHandler

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise ImportError("CatBoost is required. Add 'catboost' to requirements.txt and install.")


class CatBoostFraudTrainer:
    """
    Trains a CatBoostClassifier for fraud detection with cost-aware evaluation.

    Config keys:
      - data_path, drop_features
      - learning_rate, depth, l2_leaf_reg, iterations
      - class_weights (optional) or scale_pos_weight
      - C_FP, C_FN
      - random_seed
      - results_save_path, model_save_path
    """

    def __init__(self, config):
        self.config = config
        self.random_seed = config.get('random_seed', 42)

        # Data
        self.data_handler = CatBoostDataHandler(
            config['data_path'],
            random_seed=self.random_seed,
            drop_features=config.get('drop_features', None)
        )

        # Metrics (align with src/evaluation.py defaults unless overridden)
        self.evaluator = FraudEvaluationMetrics(
            cost_fp=config.get('C_FP', 110),
            cost_fn=config.get('C_FN', 540)
        )

        self.model = None
        self.scaler = None
        self.input_dim = None

    def prepare_data(self):
        data = self.data_handler.load_and_split()
        X_train_scaled, X_val_scaled, X_test_scaled = self.data_handler.preprocess(
            data['X_train'], data['X_val'], data['X_test']
        )
        self.input_dim = data['input_dim']
        self.scaler = self.data_handler.scaler
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test'],
        }

    def build_model(self):
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
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_val, y_val)
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False)

    def evaluate(self, X, y, split_name):
        # Predict probabilities for PR-AUC
        y_scores = self.model.predict_proba(X)[:, 1]
        # Default threshold 0.5; for cost, keep also probability outputs
        y_pred = (y_scores >= 0.5).astype(int)
        metrics = self.evaluator.calculate_metrics(y, y_pred, y_scores=y_scores)
        return metrics, y_scores, y_pred

    def visualize(self, train_metrics, val_metrics, test_metrics, y_scores_test, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PR curve on test
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(
            test_metrics['y_true'], y_scores_test
        )
        pr_auc = auc(recall, precision)
        axes[0].plot(recall, precision, label=f'PR AUC={pr_auc:.4f}')
        baseline = np.mean(test_metrics['y_true'])
        axes[0].axhline(baseline, linestyle='--', color='red', label=f'Baseline={baseline:.4f}')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Test Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bar chart of costs on splits
        splits = ['Train', 'Val', 'Test']
        costs = [train_metrics['total_cost'], val_metrics['total_cost'], test_metrics['total_cost']]
        axes[1].bar(splits, costs, color=['steelblue', 'orange', 'green'])
        axes[1].set_ylabel('Total Cost ($)')
        axes[1].set_title('Business Cost by Split')
        axes[1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        print(f"Model saved to: {save_path}")

    def run(self):
        print("=" * 60)
        print("CATBOOST FRAUD DETECTION TRAINING")
        print("=" * 60)

        data = self.prepare_data()
        self.build_model()
        self.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

        train_metrics, y_scores_train, y_pred_train = self.evaluate(data['X_train'], data['y_train'], 'train')
        val_metrics, y_scores_val, y_pred_val = self.evaluate(data['X_val'], data['y_val'], 'val')
        test_metrics, y_scores_test, y_pred_test = self.evaluate(data['X_test'], data['y_test'], 'test')

        # Attach y_true for visualization PR curve
        test_metrics_with_truth = dict(test_metrics)
        test_metrics_with_truth['y_true'] = data['y_test']

        self.visualize(
            train_metrics, val_metrics, test_metrics_with_truth,
            y_scores_test,
            self.config.get('results_save_path', 'results/figures/catboost_baseline_training.png')
        )

        self.save_model(self.config.get('model_save_path', 'models/catboost_baseline.cbm'))

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)

        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }


