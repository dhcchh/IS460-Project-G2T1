"""
CatBoost grid search tuner with results saving and plots.
"""

import os
import json
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .catboost_trainer import CatBoostFraudTrainer


class CatBoostGridSearchTuner:
    """Run grid search over configs, pick best by validation cost."""
    def __init__(self, base_config, param_grid, results_dir='results/tuning/'):
        """Store base config, grid, and output dir."""
        self.base_config = base_config
        self.param_grid = param_grid
        self.results_dir = results_dir

        # Results tracking
        self.results = []
        self.best_config = None
        self.best_cost = float('inf')
        self.best_result = None

        # Create output directory
        os.makedirs(self.results_dir, exist_ok=True)

    def get_param_combinations(self):
        """Expand parameter grid into a list of configs."""
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        all_combinations = list(product(*param_values))

        configs = []
        for params in all_combinations:
            config = self.base_config.copy()
            config.update(dict(zip(param_names, params)))
            configs.append(config)

        return configs

    def train_single_config(self, config, config_idx, total_configs):
        """Train one config and return its results dict."""
        print(f"\n{'='*70}")
        print(f"Config {config_idx}/{total_configs}: ", end="")
        print(f"lr={config.get('learning_rate')}, depth={config.get('depth')}, "
              f"l2={config.get('l2_leaf_reg')}, iter={config.get('iterations')}")
        print('='*70)

        try:
            # Create trainer for this configuration
            trainer = CatBoostFraudTrainer(config)

            # Prepare data (only once, shared across configs)
            if config_idx == 1:
                self.data = trainer.prepare_data()
            else:
                # Reuse data from first run
                trainer.input_dim = self.data_input_dim
                trainer.scaler = self.scaler

            # Build and train model
            trainer.build_model()
            trainer.train(self.data['X_train'], self.data['y_train'], 
                         self.data['X_val'], self.data['y_val'])

            # Optimize threshold
            val_probabilities = trainer.optimize_threshold(self.data['X_val'], self.data['y_val'])

            # Evaluate on test set
            test_results = trainer.evaluate(self.data['X_test'], self.data['y_test'])

            # Store data info for reuse
            if config_idx == 1:
                self.data_input_dim = trainer.input_dim
                self.scaler = trainer.scaler

            # Compile results
            result = {
                'config_idx': config_idx,
                'learning_rate': config.get('learning_rate'),
                'depth': config.get('depth'),
                'l2_leaf_reg': config.get('l2_leaf_reg'),
                'iterations': config.get('iterations'),
                'val_cost': trainer.val_cost,
                'threshold': trainer.optimal_threshold,
                'test_cost': test_results['total_cost'],
                'test_precision': test_results['precision'],
                'test_recall': test_results['recall'],
                'test_pr_auc': test_results['pr_auc'],
                'test_fp': test_results['fp'],
                'test_fn': test_results['fn'],
                'test_tp': test_results['tp'],
                'test_tn': test_results['tn']
            }

            # Print summary
            print(f"\n  Val Cost: ${result['val_cost']:,} | "
                  f"Test Cost: ${result['test_cost']:,} | "
                  f"Precision: {result['test_precision']:.3f} | "
                  f"Recall: {result['test_recall']:.3f} | "
                  f"PR-AUC: {result['test_pr_auc']:.3f}")

            # Check if this is the best model
            if result['val_cost'] < self.best_cost:
                self.best_cost = result['val_cost']
                self.best_config = config
                self.best_result = result

                # Save best model
                trainer.save_model(
                    test_results,
                    os.path.join(self.base_config.get('model_save_dir', 'models/'),
                                'catboost_best_tuned.cbm')
                )
                print(f"\n  *** NEW BEST MODEL (Val Cost: ${result['val_cost']:,}) ***")

            return result

        except Exception as e:
            print(f"\n  ERROR: {str(e)}")
            return None

    def run_grid_search(self):
        """Run all configs, save/plot results, return DataFrame."""
        print("=" * 70)
        print("CATBOOST HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 70)

        # Generate all configurations
        configs = self.get_param_combinations()

        print(f"\n[1/4] Grid Search Configuration:")
        print(f"  Total configurations: {len(configs)}")
        print(f"  Parameters:")
        for name, values in self.param_grid.items():
            print(f"    {name}: {values}")

        print(f"\n[2/4] Running grid search...")
        print("-" * 70)

        start_time = datetime.now()

        # Train all configurations
        for idx, config in enumerate(configs, 1):
            result = self.train_single_config(config, idx, len(configs))
            if result is not None:
                self.results.append(result)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "-" * 70)
        print(f"Grid search completed in {duration/60:.1f} minutes")

        results_df = pd.DataFrame(self.results)

        print(f"\n[3/4] Saving results...")
        self.save_results(results_df)

        self.print_best_config()

        print(f"\n[4/4] Generating visualizations...")
        self.visualize_results(results_df)

        print("\n" + "=" * 70)
        print("TUNING COMPLETE!")
        print("=" * 70)

        return results_df

    def save_results(self, results_df):
        """Save results to CSV and JSON."""
        # CSV
        csv_path = os.path.join(self.results_dir, 'catboost_grid_search_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"  Results CSV: {csv_path}")

        # JSON (detailed)
        json_path = os.path.join(self.results_dir, 'catboost_grid_search_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  Detailed JSON: {json_path}")

    def print_best_config(self):
        """Print best configuration details."""
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)

        print(f"\nHyperparameters:")
        print(f"  Learning Rate: {self.best_config.get('learning_rate')}")
        print(f"  Depth: {self.best_config.get('depth')}")
        print(f"  L2 Leaf Reg: {self.best_config.get('l2_leaf_reg')}")
        print(f"  Iterations: {self.best_config.get('iterations')}")

        print(f"\nValidation Performance:")
        print(f"  Validation Cost: ${self.best_result['val_cost']:,.0f}")
        print(f"  Optimal Threshold: {self.best_result['threshold']:.4f}")

        print(f"\nTest Performance (Final Evaluation Metrics):")
        print(f"  Precision: {self.best_result['test_precision']*100:.1f}%")
        print(f"  Recall: {self.best_result['test_recall']*100:.1f}%")
        print(f"  PR-AUC: {self.best_result['test_pr_auc']:.4f}")
        print(f"  Total Cost: ${self.best_result['test_cost']:,.0f}")

    def visualize_results(self, results_df):
        """Create comprehensive visualization of tuning results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('CatBoost Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

        # 1. Cost by Learning Rate
        ax = axes[0, 0]
        for depth in sorted(results_df['depth'].unique()):
            subset = results_df[results_df['depth'] == depth]
            grouped = subset.groupby('learning_rate')['test_cost'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=f'Depth={int(depth)}')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cost by Depth
        ax = axes[0, 1]
        for lr in sorted(results_df['learning_rate'].unique()):
            subset = results_df[results_df['learning_rate'] == lr]
            grouped = subset.groupby('depth')['test_cost'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=f'LR={lr}')
        ax.set_xlabel('Depth')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cost by L2 Regularization
        ax = axes[0, 2]
        grouped = results_df.groupby('l2_leaf_reg')['test_cost'].agg(['mean', 'std'])
        ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'],
               capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('L2 Leaf Regularization')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs L2 Regularization')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Precision vs Recall
        ax = axes[1, 0]
        scatter = ax.scatter(results_df['test_recall'], results_df['test_precision'],
                            c=results_df['test_cost'], cmap='RdYlGn_r', s=100, alpha=0.6)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Trade-off (color=cost)')
        plt.colorbar(scatter, ax=ax, label='Test Cost ($)')
        ax.grid(True, alpha=0.3)

        # 5. PR-AUC Distribution
        ax = axes[1, 1]
        ax.hist(results_df['test_pr_auc'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(results_df['test_pr_auc'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={results_df["test_pr_auc"].mean():.3f}')
        ax.set_xlabel('PR-AUC')
        ax.set_ylabel('Frequency')
        ax.set_title('PR-AUC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Top 10 Configurations
        ax = axes[1, 2]
        top10 = results_df.nsmallest(10, 'test_cost').copy()
        top10['config_label'] = top10.apply(
            lambda x: f"LR{x['learning_rate']:.3f}_D{int(x['depth'])}_L2{x['l2_leaf_reg']:.1f}", axis=1
        )
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top10)))
        ax.barh(range(len(top10)), top10['test_cost'], color=colors)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['config_label'], fontsize=8)
        ax.set_xlabel('Test Cost ($)')
        ax.set_title('Top 10 Configurations by Cost')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'catboost_tuning_results_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization: {save_path}")
        plt.close()

    def get_best_model_path(self):
        """Return path to best saved model."""
        return os.path.join(self.base_config.get('model_save_dir', 'models/'),
                           'catboost_best_tuned.cbm')

