"""
CatBoost Grid Search Tuner

Mirrors `VAEGridSearchTuner` with CatBoost configs.
"""

import os
import json
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd

from .catboost_trainer import CatBoostFraudTrainer


class CatBoostGridSearchTuner:
    def __init__(self, base_config, param_grid, results_dir='results/tuning/'):
        self.base_config = base_config
        self.param_grid = param_grid
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.results = []
        self.best_config = None
        self.best_cost = float('inf')
        self.best_result = None

    def get_param_combinations(self):
        names = list(self.param_grid.keys())
        values = [self.param_grid[n] for n in names]
        combos = list(product(*values))
        configs = []
        for vals in combos:
            cfg = self.base_config.copy()
            cfg.update(dict(zip(names, vals)))
            configs.append(cfg)
        return configs

    def run_grid_search(self):
        print("=" * 70)
        print("CATBOOST HYPERPARAMETER TUNING - GRID SEARCH")
        print("=" * 70)

        configs = self.get_param_combinations()
        print(f"Total configurations: {len(configs)}")

        start_time = datetime.now()
        for idx, cfg in enumerate(configs, 1):
            print(f"\n--- Config {idx}/{len(configs)}: {cfg}")
            trainer = CatBoostFraudTrainer(cfg)
            data = trainer.prepare_data() if idx == 1 else data  # reuse data from first run
            trainer.build_model()
            trainer.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
            train_metrics, _, _ = trainer.evaluate(data['X_train'], data['y_train'], 'train')
            val_metrics, _, _ = trainer.evaluate(data['X_val'], data['y_val'], 'val')
            test_metrics, _, _ = trainer.evaluate(data['X_test'], data['y_test'], 'test')

            result = {
                'config_idx': idx,
                'learning_rate': cfg.get('learning_rate'),
                'depth': cfg.get('depth'),
                'l2_leaf_reg': cfg.get('l2_leaf_reg'),
                'iterations': cfg.get('iterations'),
                'train_cost': train_metrics['total_cost'],
                'val_cost': val_metrics['total_cost'],
                'test_cost': test_metrics['total_cost'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_pr_auc': test_metrics['pr_auc'],
            }
            print(f"Val Cost: ${result['val_cost']:,} | Test Cost: ${result['test_cost']:,}")

            if result['val_cost'] < self.best_cost:
                self.best_cost = result['val_cost']
                self.best_config = cfg
                self.best_result = result

            self.results.append(result)

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nGrid search completed in {duration/60:.1f} minutes")

        results_df = pd.DataFrame(self.results)
        self.save_results(results_df)
        self.print_best_config()
        return results_df

    def save_results(self, results_df):
        csv_path = os.path.join(self.results_dir, 'catboost_grid_search_results.csv')
        results_df.to_csv(csv_path, index=False)
        json_path = os.path.join(self.results_dir, 'catboost_grid_search_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved CSV: {csv_path}")
        print(f"Saved JSON: {json_path}")

    def print_best_config(self):
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION (by Val Cost)")
        print("=" * 70)
        print(self.best_config)
        print(f"Val Cost: ${self.best_cost:,.0f}")
        print(f"Test Cost: ${self.best_result['test_cost']:,.0f}")


