"""
VAE Hyperparameter Tuner Class

Extends VAEFraudTrainer to perform grid search over hyperparameters.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime
import json
import os

from .vae_trainer import VAEFraudTrainer


class VAEGridSearchTuner:
    """
    Performs grid search hyperparameter tuning for VAE fraud detection.

    Responsibilities:
    - Define parameter grid
    - Train multiple VAE configurations
    - Track and compare results
    - Save best model
    - Visualize tuning results

    No overlap with VAEFraudTrainer - uses it as a component.
    """

    def __init__(self, base_config, param_grid, results_dir='results/tuning/'):
        """
        Initialize tuner with base configuration and search grid.

        Args:
            base_config: Base configuration dictionary
            param_grid: Dictionary of parameter lists to search
            results_dir: Directory to save tuning results
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.results_dir = results_dir

        # Results tracking
        self.results = []
        self.best_config = None
        self.best_cost = float('inf')

        # Create output directory
        os.makedirs(self.results_dir, exist_ok=True)

    def get_param_combinations(self):
        """
        Generate all parameter combinations from grid.

        Returns:
            List of configuration dictionaries
        """
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
        """
        Train a single VAE configuration.

        Args:
            config: Configuration dictionary
            config_idx: Current configuration index
            total_configs: Total number of configurations

        Returns:
            Dictionary with configuration and results
        """
        print(f"\n{'='*70}")
        print(f"Config {config_idx}/{total_configs}: ", end="")
        print(f"hidden={config['hidden_dim']}, latent={config['latent_dim']}, "
              f"beta={config['beta']}, lr={config['learning_rate']}")
        print('='*70)

        try:
            # Create trainer for this configuration
            trainer = VAEFraudTrainer(config)

            # Prepare data (only once, shared across configs)
            if config_idx == 1:
                self.data = trainer.prepare_data()
            else:
                # Reuse data from first run
                trainer.input_dim = self.data_input_dim
                trainer.scaler = self.scaler

            # Build and train model
            trainer.build_model()
            trainer.train(self.data['X_train'], self.data['X_val'], self.data['y_val'])

            # Optimize threshold
            val_errors = trainer.optimize_threshold(self.data['X_val'], self.data['y_val'])

            # Evaluate on test set
            test_results = trainer.evaluate(self.data['X_test'], self.data['y_test'])

            # Store data info for reuse
            if config_idx == 1:
                self.data_input_dim = trainer.input_dim
                self.scaler = trainer.scaler

            # Compile results
            result = {
                'config_idx': config_idx,
                'hidden_dim': config['hidden_dim'],
                'latent_dim': config['latent_dim'],
                'beta': config['beta'],
                'learning_rate': config['learning_rate'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'train_loss': trainer.train_losses[-1],
                'val_loss': trainer.val_losses[-1],
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
                                'vae_best_tuned.pth')
                )
                print(f"\n  *** NEW BEST MODEL (Val Cost: ${result['val_cost']:,}) ***")

            return result

        except Exception as e:
            print(f"\n  ERROR: {str(e)}")
            return None

    def run_grid_search(self):
        """
        Execute grid search over all parameter combinations.

        Returns:
            DataFrame with all results
        """
        print("=" * 70)
        print("VAE HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 70)

        # Generate all configurations
        configs = self.get_param_combinations()

        print(f"\n[1/4] Grid Search Configuration:")
        print(f"  Total configurations: {len(configs)}")
        print(f"  Device: {self.base_config['device']}")
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
        csv_path = os.path.join(self.results_dir, 'grid_search_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"  Results CSV: {csv_path}")

        # JSON (detailed)
        json_path = os.path.join(self.results_dir, 'grid_search_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  Detailed JSON: {json_path}")

    def print_best_config(self):
        """Print best configuration details."""
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)

        print(f"\nHyperparameters:")
        print(f"  Hidden Dimension: {self.best_config['hidden_dim']}")
        print(f"  Latent Dimension: {self.best_config['latent_dim']}")
        print(f"  Beta: {self.best_config['beta']}")
        print(f"  Learning Rate: {self.best_config['learning_rate']}")
        print(f"  Epochs: {self.best_config['epochs']}")

        print(f"\nValidation Performance:")
        print(f"  Validation Cost: ${self.best_result['val_cost']:,.0f}")
        print(f"  Optimal Threshold: {self.best_result['threshold']:.6f}")

        print(f"\nTest Performance (Final Evaluation Metrics):")
        print(f"  Precision: {self.best_result['test_precision']*100:.1f}%")
        print(f"  Recall: {self.best_result['test_recall']*100:.1f}%")
        print(f"  PR-AUC: {self.best_result['test_pr_auc']:.4f}")
        print(f"  Total Cost: ${self.best_result['test_cost']:,.0f}")

    def visualize_results(self, results_df):
        """Create comprehensive visualization of tuning results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('VAE Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

        # 1. Cost by Hidden Dim
        ax = axes[0, 0]
        for beta in sorted(results_df['beta'].unique()):
            subset = results_df[results_df['beta'] == beta]
            grouped = subset.groupby('hidden_dim')['test_cost'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=f'β={beta}')
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs Hidden Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Cost by Latent Dim
        ax = axes[0, 1]
        for beta in sorted(results_df['beta'].unique()):
            subset = results_df[results_df['beta'] == beta]
            grouped = subset.groupby('latent_dim')['test_cost'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=f'β={beta}')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs Latent Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cost by Beta
        ax = axes[0, 2]
        grouped = results_df.groupby('beta')['test_cost'].agg(['mean', 'std'])
        ax.bar(grouped.index.astype(str), grouped['mean'], yerr=grouped['std'],
               capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('Beta (KL Weight)')
        ax.set_ylabel('Test Cost ($)')
        ax.set_title('Cost vs Beta')
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
            lambda x: f"H{int(x['hidden_dim'])}_L{int(x['latent_dim'])}_β{x['beta']}", axis=1
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
        save_path = os.path.join(self.results_dir, 'tuning_results_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization: {save_path}")
        plt.close()

    def get_best_model_path(self):
        """Return path to best saved model."""
        return os.path.join(self.base_config.get('model_save_dir', 'models/'),
                           'vae_best_tuned.pth')
