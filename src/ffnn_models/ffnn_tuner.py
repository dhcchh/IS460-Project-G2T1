"""
FFNN Hyperparameter Tuner Class

Performs grid search hyperparameter tuning for FFNN fraud detection.
Mirrors VAEGridSearchTuner and CatBoostGridSearchTuner for consistency.
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

from .ffnn_trainer import FFNNFraudTrainer


class FFNNGridSearchTuner:
    """
    Performs grid search hyperparameter tuning for FFNN fraud detection.
    
    Responsibilities:
    - Define parameter grid
    - Train multiple FFNN configurations
    - Track and compare results
    - Save best model
    - Visualize tuning results
    
    Uses FFNNFraudTrainer as a component for each configuration.
    """
    
    def __init__(self, base_config, param_grid, results_dir='results/tuning/'):
        """
        Initialize tuner with base configuration and search grid.
        
        Args:
            base_config: Base configuration dictionary (shared across all configs)
            param_grid: Dictionary of parameter lists to search over
                Example: {
                    'hidden_dim': [64, 128, 256],
                    'dropout_rate': [0.1, 0.2, 0.3],
                    'learning_rate': [0.0001, 0.001, 0.01]
                }
            results_dir: Directory to save tuning results
        """
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
        os.makedirs(self.base_config.get('model_save_dir', 'models/'), exist_ok=True)
    
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
        Train a single FFNN configuration.
        
        Args:
            config: Configuration dictionary
            config_idx: Current configuration index (1-indexed)
            total_configs: Total number of configurations
            
        Returns:
            Dictionary with configuration and results, or None if training failed
        """
        print(f"\n{'='*70}")
        print(f"Config {config_idx}/{total_configs}: ", end="")
        print(f"hidden={config['hidden_dim']}, dropout={config['dropout_rate']}, "
              f"lr={config['learning_rate']}, batch={config.get('batch_size', 64)}")
        print('='*70)
        
        try:
            # Update save paths for this config
            config['model_save_path'] = os.path.join(
                self.base_config.get('model_save_dir', 'models/'),
                f'ffnn_config_{config_idx}.pth'
            )
            config['results_save_path'] = os.path.join(
                self.results_dir,
                f'ffnn_config_{config_idx}_viz.png'
            )
            
            # Create trainer for this configuration
            trainer = FFNNFraudTrainer(config)
            
            # Run training pipeline
            test_results = trainer.run()
            
            # Compile results
            result = {
                'config_idx': config_idx,
                'hidden_dim': config['hidden_dim'],
                'dropout_rate': config['dropout_rate'],
                'learning_rate': config['learning_rate'],
                'batch_size': config.get('batch_size', 64),
                'epochs': config.get('epochs', 100),
                'use_class_weights': config.get('use_class_weights', True),
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
                'test_tn': test_results['tn'],
                'savings': test_results['savings']
            }
            
            # Print summary
            print(f"\n  Val Cost: ${result['val_cost']:,} | "
                  f"Test Cost: ${result['test_cost']:,} | "
                  f"Precision: {result['test_precision']:.3f} | "
                  f"Recall: {result['test_recall']:.3f} | "
                  f"PR-AUC: {result['test_pr_auc']:.3f}")
            
            # Check if this is the best model (using validation cost)
            if result['val_cost'] < self.best_cost:
                self.best_cost = result['val_cost']
                self.best_config = config
                self.best_result = result
                
                # Copy best model
                import shutil
                best_model_path = os.path.join(
                    self.base_config.get('model_save_dir', 'models/'),
                    'ffnn_best_tuned.pth'
                )
                shutil.copy(config['model_save_path'], best_model_path)
                
                print(f"\n  *** NEW BEST MODEL (Val Cost: ${result['val_cost']:,}) ***")
            
            return result
        
        except Exception as e:
            print(f"\n  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_grid_search(self):
        """
        Execute grid search over all parameter combinations.
        
        Returns:
            DataFrame with all results
        """
        print("=" * 70)
        print("FFNN HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 70)
        
        # Generate all configurations
        configs = self.get_param_combinations()
        
        print(f"\n[1/4] Grid Search Configuration:")
        print(f"  Total configurations: {len(configs)}")
        print(f"  Device: {self.base_config.get('device', 'cpu')}")
        print(f"  Data path: {self.base_config['data_path']}")
        print(f"  Parameters to tune:")
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
        print(f"Successfully trained: {len(self.results)}/{len(configs)} configurations")
        
        if len(self.results) == 0:
            print("\nERROR: No configurations completed successfully!")
            return pd.DataFrame()
        
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
        """
        Save results to CSV and JSON.
        
        Args:
            results_df: DataFrame with all tuning results
        """
        # CSV
        csv_path = os.path.join(self.results_dir, 'ffnn_grid_search_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"  Results CSV: {csv_path}")
        
        # JSON (detailed)
        json_path = os.path.join(self.results_dir, 'ffnn_grid_search_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  Detailed JSON: {json_path}")
        
        # Summary statistics
        summary_path = os.path.join(self.results_dir, 'ffnn_grid_search_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("FFNN Grid Search Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total configurations: {len(results_df)}\n")
            f.write(f"Best validation cost: ${self.best_cost:,.0f}\n")
            f.write(f"Best test cost: ${self.best_result['test_cost']:,.0f}\n\n")
            f.write("Cost Statistics:\n")
            f.write(f"  Mean: ${results_df['test_cost'].mean():,.0f}\n")
            f.write(f"  Std: ${results_df['test_cost'].std():,.0f}\n")
            f.write(f"  Min: ${results_df['test_cost'].min():,.0f}\n")
            f.write(f"  Max: ${results_df['test_cost'].max():,.0f}\n\n")
            f.write("Best Configuration:\n")
            for key, value in self.best_result.items():
                f.write(f"  {key}: {value}\n")
        print(f"  Summary: {summary_path}")
    
    def print_best_config(self):
        """Print best configuration details."""
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        
        print(f"\nHyperparameters:")
        print(f"  Hidden Dimension: {self.best_config['hidden_dim']}")
        print(f"  Dropout Rate: {self.best_config['dropout_rate']}")
        print(f"  Learning Rate: {self.best_config['learning_rate']}")
        print(f"  Batch Size: {self.best_config.get('batch_size', 64)}")
        print(f"  Epochs: {self.best_config.get('epochs', 100)}")
        print(f"  Use Class Weights: {self.best_config.get('use_class_weights', True)}")
        
        print(f"\nValidation Performance:")
        print(f"  Validation Cost: ${self.best_result['val_cost']:,.0f}")
        print(f"  Optimal Threshold: {self.best_result['threshold']:.3f}")
        
        print(f"\nTest Performance (Final Evaluation Metrics):")
        print(f"  Precision: {self.best_result['test_precision']*100:.1f}%")
        print(f"  Recall: {self.best_result['test_recall']*100:.1f}%")
        print(f"  PR-AUC: {self.best_result['test_pr_auc']:.4f}")
        print(f"  Total Cost: ${self.best_result['test_cost']:,.0f}")
        print(f"  Net Savings: ${self.best_result['savings']:,.0f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {self.best_result['test_tp']}")
        print(f"  True Negatives: {self.best_result['test_tn']}")
        print(f"  False Positives: {self.best_result['test_fp']}")
        print(f"  False Negatives: {self.best_result['test_fn']}")
    
    def visualize_results(self, results_df):
        """
        Create comprehensive visualization of tuning results.
        
        Args:
            results_df: DataFrame with all tuning results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('FFNN Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # 1. Cost by Hidden Dimension
        ax = axes[0, 0]
        if 'hidden_dim' in self.param_grid and len(self.param_grid['hidden_dim']) > 1:
            for lr in sorted(results_df['learning_rate'].unique()):
                subset = results_df[results_df['learning_rate'] == lr]
                grouped = subset.groupby('hidden_dim')['test_cost'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f'LR={lr}')
            ax.set_xlabel('Hidden Dimension')
            ax.set_ylabel('Test Cost ($)')
            ax.set_title('Cost vs Hidden Dimension')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Hidden dimension\nnot tuned', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 2. Cost by Dropout Rate
        ax = axes[0, 1]
        if 'dropout_rate' in self.param_grid and len(self.param_grid['dropout_rate']) > 1:
            for lr in sorted(results_df['learning_rate'].unique()):
                subset = results_df[results_df['learning_rate'] == lr]
                grouped = subset.groupby('dropout_rate')['test_cost'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f'LR={lr}')
            ax.set_xlabel('Dropout Rate')
            ax.set_ylabel('Test Cost ($)')
            ax.set_title('Cost vs Dropout Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Dropout rate\nnot tuned', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 3. Cost by Learning Rate
        ax = axes[0, 2]
        if 'learning_rate' in self.param_grid and len(self.param_grid['learning_rate']) > 1:
            grouped = results_df.groupby('learning_rate')['test_cost'].agg(['mean', 'std'])
            ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'],
                   capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels([f'{lr:.4f}' for lr in grouped.index])
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Test Cost ($)')
            ax.set_title('Cost vs Learning Rate')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Learning rate\nnot tuned', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 4. Precision vs Recall Trade-off
        ax = axes[1, 0]
        scatter = ax.scatter(results_df['test_recall'], results_df['test_precision'],
                            c=results_df['test_cost'], cmap='RdYlGn_r', s=100, alpha=0.6)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Trade-off (color=cost)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Test Cost ($)')
        ax.grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = results_df['val_cost'].idxmin()
        ax.scatter(results_df.loc[best_idx, 'test_recall'], 
                  results_df.loc[best_idx, 'test_precision'],
                  s=200, marker='*', c='gold', edgecolors='black', 
                  linewidths=2, label='Best Model', zorder=5)
        ax.legend()
        
        # 5. PR-AUC Distribution
        ax = axes[1, 1]
        ax.hist(results_df['test_pr_auc'], bins=min(20, len(results_df)), 
               edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(results_df['test_pr_auc'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={results_df["test_pr_auc"].mean():.3f}')
        ax.axvline(self.best_result['test_pr_auc'], color='green', linestyle='--',
                   linewidth=2, label=f'Best={self.best_result["test_pr_auc"]:.3f}')
        ax.set_xlabel('PR-AUC')
        ax.set_ylabel('Frequency')
        ax.set_title('PR-AUC Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Top 10 Configurations
        ax = axes[1, 2]
        top10 = results_df.nsmallest(min(10, len(results_df)), 'test_cost').copy()
        top10['config_label'] = top10.apply(
            lambda x: f"H{int(x['hidden_dim'])}_D{x['dropout_rate']:.2f}_LR{x['learning_rate']:.4f}", 
            axis=1
        )
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top10)))
        ax.barh(range(len(top10)), top10['test_cost'], color=colors)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['config_label'], fontsize=8)
        ax.set_xlabel('Test Cost ($)')
        ax.set_title(f'Top {len(top10)} Configurations by Cost')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'ffnn_tuning_results_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization: {save_path}")
        plt.close()
        
        # Additional heatmap if we have multiple dimensions
        if len(self.param_grid) >= 2:
            self._create_heatmap(results_df)
    
    def _create_heatmap(self, results_df):
        """
        Create heatmap showing interaction between parameters.
        
        Args:
            results_df: DataFrame with all tuning results
        """
        # Find two most varied parameters
        param_names = list(self.param_grid.keys())
        
        if len(param_names) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Heatmap 1: First two parameters
            param1, param2 = param_names[0], param_names[1]
            pivot = results_df.pivot_table(
                values='test_cost', 
                index=param2, 
                columns=param1, 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                       ax=axes[0], cbar_kws={'label': 'Test Cost ($)'})
            axes[0].set_title(f'Cost Heatmap: {param1} vs {param2}')
            
            # Heatmap 2: PR-AUC
            pivot_auc = results_df.pivot_table(
                values='test_pr_auc', 
                index=param2, 
                columns=param1, 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[1], cbar_kws={'label': 'PR-AUC'})
            axes[1].set_title(f'PR-AUC Heatmap: {param1} vs {param2}')
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, 'ffnn_tuning_heatmaps.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Heatmap: {save_path}")
            plt.close()
    
    def get_best_model_path(self):
        """
        Return path to best saved model.
        
        Returns:
            Path to best model checkpoint
        """
        return os.path.join(
            self.base_config.get('model_save_dir', 'models/'),
            'ffnn_best_tuned.pth'
        )
    
    def get_best_config(self):
        """
        Return best configuration dictionary.
        
        Returns:
            Dictionary with best hyperparameters
        """
        return self.best_config
    
    def get_results_dataframe(self):
        """
        Return results as DataFrame.
        
        Returns:
            DataFrame with all tuning results
        """
        return pd.DataFrame(self.results)