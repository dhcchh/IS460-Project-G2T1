"""
Ensemble Module for Fraud Detection

Combines CatBoost, VAE, and FFNN models with cost-based weighting.
Lower cost models receive higher weights in the ensemble.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from evaluation import FraudEvaluationMetrics
from catboost_models.catboost_base import compute_prediction_scores as catboost_scores
from vae_models.vae_base import VAE, compute_reconstruction_errors
from ffnn_models.ffnn_base import FraudDetectionFFNN, compute_prediction_scores as ffnn_scores

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("CatBoost is required. Install with: pip install catboost")


class FraudEnsemble:
    """
    Ensemble of CatBoost, VAE, and FFNN models for fraud detection.

    Weights models based on inverse cost - models with lower cost get higher weight.
    """

    def __init__(
        self,
        catboost_model_path='models/catboost_best_tuned.cbm',
        vae_model_path='models/vae_best_tuned.pth',
        ffnn_model_path='models/ffnn_best_tuned.pth',
        data_path=None,
        C_FP=550,
        C_FN=110,
        device='cpu'
    ):
        """
        Initialize ensemble with model paths.

        Args:
            catboost_model_path: Path to CatBoost .cbm file
            vae_model_path: Path to VAE .pth file
            ffnn_model_path: Path to FFNN .pth file
            data_path: Path to CSV data file
            C_FP: Cost per false positive
            C_FN: Cost per false negative
            device: 'cuda' or 'cpu' for PyTorch models
        """
        self.catboost_model_path = catboost_model_path
        self.vae_model_path = vae_model_path
        self.ffnn_model_path = ffnn_model_path
        self.data_path = data_path
        self.C_FP = C_FP
        self.C_FN = C_FN
        self.device = device

        # Model components
        self.catboost_model = None
        self.catboost_metadata = None
        self.vae_model = None
        self.vae_metadata = None
        self.ffnn_model = None
        self.ffnn_metadata = None

        # Weights and thresholds
        self.catboost_weight = None
        self.vae_weight = None
        self.ffnn_weight = None
        self.catboost_threshold = None
        self.vae_threshold = None
        self.ffnn_threshold = None

        # Data
        self.X_catboost = None  # Data scaled with CatBoost scaler
        self.X_vae = None  # Data scaled with VAE scaler
        self.X_ffnn = None  # Data scaled with FFNN scaler
        self.y = None
        self.catboost_scaler = None
        self.vae_scaler = None
        self.ffnn_scaler = None

    def load_models(self):
        """Load CatBoost, VAE, and FFNN models from disk."""
        print("=" * 60)
        print("LOADING MODELS")
        print("=" * 60)

        # Load CatBoost
        print(f"\n[1/3] Loading CatBoost model...")
        if not os.path.exists(self.catboost_model_path):
            raise FileNotFoundError(f"CatBoost model not found: {self.catboost_model_path}")

        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(self.catboost_model_path)

        # Load CatBoost metadata
        metadata_path = self.catboost_model_path.replace('.cbm', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.catboost_metadata = pickle.load(f)
            self.catboost_threshold = self.catboost_metadata['optimal_threshold']
            print(f"  CatBoost loaded successfully")
            print(f"  Optimal threshold: {self.catboost_threshold:.4f}")
            print(f"  Test cost: ${self.catboost_metadata['metrics']['test_cost']:,.0f}")
        else:
            print(f"  Warning: Metadata file not found, using default threshold 0.5")
            self.catboost_threshold = 0.5

        # Load VAE
        print(f"\n[2/3] Loading VAE model...")
        if not os.path.exists(self.vae_model_path):
            raise FileNotFoundError(f"VAE model not found: {self.vae_model_path}")

        checkpoint = torch.load(self.vae_model_path, map_location=self.device, weights_only=False)
        self.vae_metadata = checkpoint

        # Reconstruct VAE architecture from saved config
        config = checkpoint['config']
        self.vae_model = VAE(
            config['input_dim'],
            config['hidden_dim'],
            config['latent_dim']
        ).to(self.device)
        self.vae_model.load_state_dict(checkpoint['model_state_dict'])
        self.vae_model.eval()

        self.vae_threshold = checkpoint['optimal_threshold']
        print(f"  VAE loaded successfully")
        print(f"  Optimal threshold: {self.vae_threshold:.6f}")
        print(f"  Test cost: ${checkpoint['metrics']['test_cost']:,.0f}")

        # Load FFNN
        print(f"\n[3/3] Loading FFNN model...")
        if not os.path.exists(self.ffnn_model_path):
            raise FileNotFoundError(f"FFNN model not found: {self.ffnn_model_path}")

        checkpoint = torch.load(self.ffnn_model_path, map_location=self.device, weights_only=False)
        self.ffnn_metadata = checkpoint

        # Reconstruct FFNN architecture from saved config
        config = checkpoint['config']
        self.ffnn_model = FraudDetectionFFNN(
            config['input_dim'],
            config['hidden_dim'],
            config.get('dropout_rate', 0.2)
        ).to(self.device)
        self.ffnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.ffnn_model.eval()

        self.ffnn_threshold = checkpoint['optimal_threshold']
        print(f"  FFNN loaded successfully")
        print(f"  Optimal threshold: {self.ffnn_threshold:.4f}")
        print(f"  Test cost: ${checkpoint['metrics']['test_cost']:,.0f}")

        print("\nAll models loaded successfully!")

    def load_data(self):
        """Load and preprocess the entire dataset."""
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        if self.data_path is None:
            raise ValueError("data_path must be provided")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"\nLoading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Extract labels
        self.y = df['Class'].values

        # Drop features to match training (logreg_baseline feature set)
        # Both CatBoost and VAE were trained with these features dropped:
        # V13, V15, V19, V20, V22, V23, V24, V25, V26, Time, Amount, Hour, Hour_sin, log_Amount
        # This leaves 22 features: V1-V12, V14, V16-V18, V21, V27-V28, Day, Time_bin, Amount_bin
        drop_features = [
            'Class',  # Target variable
            'V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',  # Low importance V features
            'Time', 'Amount', 'Hour', 'Hour_sin', 'log_Amount'  # Replaced by engineered features
        ]

        print(f"\nDropping features to match training:")
        print(f"  Features to drop: {', '.join(drop_features)}")

        # Get features
        X_raw = df.drop(columns=drop_features).values

        # Load scalers from all models
        # IMPORTANT: Each model has its own scaler!
        # - VAE scaler: Fitted on ONLY normal transactions
        # - CatBoost scaler: Fitted on ALL data (normal + fraud)
        # - FFNN scaler: Fitted on ALL data (normal + fraud)
        if self.catboost_metadata and 'scaler' in self.catboost_metadata:
            self.catboost_scaler = self.catboost_metadata['scaler']
            print(f"  Loaded CatBoost scaler (fitted on all data)")
        else:
            raise ValueError("CatBoost scaler not found in metadata!")

        if self.vae_metadata and 'scaler' in self.vae_metadata:
            self.vae_scaler = self.vae_metadata['scaler']
            print(f"  Loaded VAE scaler (fitted on normal transactions only)")
        else:
            raise ValueError("VAE scaler not found in metadata!")

        if self.ffnn_metadata and 'scaler' in self.ffnn_metadata:
            self.ffnn_scaler = self.ffnn_metadata['scaler']
            print(f"  Loaded FFNN scaler (fitted on all data)")
        else:
            raise ValueError("FFNN scaler not found in metadata!")

        # Scale data with ALL scalers
        # Each model needs data scaled the same way it was trained
        self.X_catboost = self.catboost_scaler.transform(X_raw)
        self.X_vae = self.vae_scaler.transform(X_raw)
        self.X_ffnn = self.ffnn_scaler.transform(X_raw)

        print(f"\nDataset loaded and scaled:")
        print(f"  Total samples: {len(X_raw):,}")
        print(f"  Features: {X_raw.shape[1]} (logreg_baseline feature set)")
        print(f"  Normal transactions: {(self.y == 0).sum():,} ({(self.y == 0).sum() / len(self.y) * 100:.2f}%)")
        print(f"  Fraud transactions: {(self.y == 1).sum():,} ({(self.y == 1).sum() / len(self.y) * 100:.2f}%)")
        print(f"\n  Note: Data scaled separately for each model using their respective scalers")
        print(f"\n  ⚠️  IMPORTANT: Ensemble evaluates on FULL dataset ({(self.y == 1).sum()} fraud)")
        print(f"      Individual models were trained/tested on smaller subsets:")
        print(f"      - CatBoost test set: ~20% of fraud (~98 cases)")
        print(f"      - VAE test set: ~50% of fraud (~246 cases)")
        print(f"      Cost values will be HIGHER here due to more fraud cases!")

    def get_model_predictions(self):
        """Get predictions and scores from all models using their respective scalers."""
        print("\n" + "=" * 60)
        print("GENERATING MODEL PREDICTIONS")
        print("=" * 60)

        # CatBoost predictions (using CatBoost-scaled data)
        print("\n[1/3] Getting CatBoost predictions...")
        print(f"  Using CatBoost scaler (fitted on all data)")
        catboost_probs = catboost_scores(self.catboost_model, self.X_catboost)
        catboost_preds = (catboost_probs >= self.catboost_threshold).astype(int)
        print(f"  CatBoost predictions: {catboost_preds.sum():,} flagged as fraud")

        # VAE predictions (using VAE-scaled data)
        print("\n[2/3] Getting VAE predictions...")
        print(f"  Using VAE scaler (fitted on normal transactions only)")
        X_vae_tensor = torch.FloatTensor(self.X_vae).to(self.device)
        vae_errors = compute_reconstruction_errors(self.vae_model, X_vae_tensor)
        vae_preds = (vae_errors > self.vae_threshold).astype(int)
        print(f"  VAE predictions: {vae_preds.sum():,} flagged as fraud")

        # FFNN predictions (using FFNN-scaled data)
        print("\n[3/3] Getting FFNN predictions...")
        print(f"  Using FFNN scaler (fitted on all data)")
        X_ffnn_tensor = torch.FloatTensor(self.X_ffnn).to(self.device)
        ffnn_probs = ffnn_scores(self.ffnn_model, X_ffnn_tensor, device=self.device)
        ffnn_preds = (ffnn_probs >= self.ffnn_threshold).astype(int)
        print(f"  FFNN predictions: {ffnn_preds.sum():,} flagged as fraud")

        return {
            'catboost_probs': catboost_probs,
            'catboost_preds': catboost_preds,
            'vae_errors': vae_errors,
            'vae_preds': vae_preds,
            'ffnn_probs': ffnn_probs,
            'ffnn_preds': ffnn_preds
        }

    def compute_individual_costs(self, predictions):
        """Compute cost for each individual model."""
        print("\n" + "=" * 60)
        print("COMPUTING INDIVIDUAL MODEL COSTS")
        print("=" * 60)

        evaluator = FraudEvaluationMetrics(cost_fp=self.C_FP, cost_fn=self.C_FN)

        # CatBoost cost
        print("\n[1/3] CatBoost Performance:")
        catboost_metrics = evaluator.calculate_metrics(
            self.y,
            predictions['catboost_preds'],
            y_scores=predictions['catboost_probs']
        )
        catboost_cost = catboost_metrics['total_cost']

        print(f"  Precision: {catboost_metrics['precision']:.4f}")
        print(f"  Recall: {catboost_metrics['recall']:.4f}")
        print(f"  PR-AUC: {catboost_metrics['pr_auc']:.4f}" if catboost_metrics['pr_auc'] else "  PR-AUC: N/A")
        print(f"  Total Cost: ${catboost_cost:,.0f}")
        print(f"  False Positives: {catboost_metrics['false_positives']:,}")
        print(f"  False Negatives: {catboost_metrics['false_negatives']:,}")

        # VAE cost
        print("\n[2/3] VAE Performance:")
        vae_metrics = evaluator.calculate_metrics(
            self.y,
            predictions['vae_preds'],
            y_scores=predictions['vae_errors']
        )
        vae_cost = vae_metrics['total_cost']

        print(f"  Precision: {vae_metrics['precision']:.4f}")
        print(f"  Recall: {vae_metrics['recall']:.4f}")
        print(f"  PR-AUC: {vae_metrics['pr_auc']:.4f}" if vae_metrics['pr_auc'] else "  PR-AUC: N/A")
        print(f"  Total Cost: ${vae_cost:,.0f}")
        print(f"  False Positives: {vae_metrics['false_positives']:,}")
        print(f"  False Negatives: {vae_metrics['false_negatives']:,}")

        # FFNN cost
        print("\n[3/3] FFNN Performance:")
        ffnn_metrics = evaluator.calculate_metrics(
            self.y,
            predictions['ffnn_preds'],
            y_scores=predictions['ffnn_probs']
        )
        ffnn_cost = ffnn_metrics['total_cost']

        print(f"  Precision: {ffnn_metrics['precision']:.4f}")
        print(f"  Recall: {ffnn_metrics['recall']:.4f}")
        print(f"  PR-AUC: {ffnn_metrics['pr_auc']:.4f}" if ffnn_metrics['pr_auc'] else "  PR-AUC: N/A")
        print(f"  Total Cost: ${ffnn_cost:,.0f}")
        print(f"  False Positives: {ffnn_metrics['false_positives']:,}")
        print(f"  False Negatives: {ffnn_metrics['false_negatives']:,}")

        return {
            'catboost': catboost_metrics,
            'vae': vae_metrics,
            'ffnn': ffnn_metrics,
            'catboost_cost': catboost_cost,
            'vae_cost': vae_cost,
            'ffnn_cost': ffnn_cost
        }

    def compute_ensemble_weights(self, costs):
        """
        Compute ensemble weights based on normalized cost per fraud case.

        This approach is FAIR because it accounts for:
        1. Different training paradigms (CatBoost/FFNN supervised, VAE unsupervised)
        2. CatBoost/FFNN saw fraud cases during training; VAE saw 0
        3. All evaluated on same full dataset (492 fraud cases)

        Normalizing by fraud cases ensures fair comparison regardless of model type.
        Lower cost per fraud case = higher weight.
        """
        print("\n" + "=" * 60)
        print("COMPUTING ENSEMBLE WEIGHTS (FAIR NORMALIZED APPROACH)")
        print("=" * 60)

        catboost_cost = costs['catboost_cost']
        vae_cost = costs['vae_cost']
        ffnn_cost = costs['ffnn_cost']

        # Number of fraud cases in full dataset
        n_fraud = self.y.sum()

        # Normalize costs by number of fraud cases
        # This makes comparison fair between supervised (CatBoost/FFNN) and unsupervised (VAE) models
        catboost_cost_per_fraud = catboost_cost / n_fraud
        vae_cost_per_fraud = vae_cost / n_fraud
        ffnn_cost_per_fraud = ffnn_cost / n_fraud

        print(f"\nRaw costs on full dataset ({n_fraud} fraud cases):")
        print(f"  CatBoost total cost: ${catboost_cost:,.0f}")
        print(f"  VAE total cost: ${vae_cost:,.0f}")
        print(f"  FFNN total cost: ${ffnn_cost:,.0f}")

        print(f"\nNormalized cost per fraud case:")
        print(f"  CatBoost: ${catboost_cost_per_fraud:.2f} per fraud case")
        print(f"  VAE: ${vae_cost_per_fraud:.2f} per fraud case")
        print(f"  FFNN: ${ffnn_cost_per_fraud:.2f} per fraud case")

        # Inverse normalized cost weighting
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        inverse_catboost = 1.0 / (catboost_cost_per_fraud + epsilon)
        inverse_vae = 1.0 / (vae_cost_per_fraud + epsilon)
        inverse_ffnn = 1.0 / (ffnn_cost_per_fraud + epsilon)

        # Normalize to sum to 1
        total_inverse = inverse_catboost + inverse_vae + inverse_ffnn
        self.catboost_weight = inverse_catboost / total_inverse
        self.vae_weight = inverse_vae / total_inverse
        self.ffnn_weight = inverse_ffnn / total_inverse

        print(f"\nFair ensemble weights (based on normalized cost):")
        print(f"  CatBoost: {self.catboost_weight:.4f} (${catboost_cost_per_fraud:.2f}/fraud)")
        print(f"  VAE: {self.vae_weight:.4f} (${vae_cost_per_fraud:.2f}/fraud)")
        print(f"  FFNN: {self.ffnn_weight:.4f} (${ffnn_cost_per_fraud:.2f}/fraud)")

        print(f"\n  ℹ️  Weighting Fairness Notes:")
        print(f"      - CatBoost (supervised): Trained on fraud examples")
        print(f"      - VAE (unsupervised): Trained on 0 fraud examples")
        print(f"      - FFNN (supervised): Trained on fraud examples")
        print(f"      - Normalized cost accounts for different training paradigms")
        print(f"      - All evaluated on same full dataset ({n_fraud} fraud cases)")

        # Cost savings compared to baseline
        baseline_cost = self.y.sum() * self.C_FN
        catboost_savings = baseline_cost - catboost_cost
        vae_savings = baseline_cost - vae_cost
        ffnn_savings = baseline_cost - ffnn_cost

        print(f"\nCost savings vs. baseline (no detection):")
        print(f"  Baseline cost: ${baseline_cost:,.0f}")
        print(f"  CatBoost savings: ${catboost_savings:,.0f} ({catboost_savings/baseline_cost*100:.2f}%)")
        print(f"  VAE savings: ${vae_savings:,.0f} ({vae_savings/baseline_cost*100:.2f}%)")
        print(f"  FFNN savings: ${ffnn_savings:,.0f} ({ffnn_savings/baseline_cost*100:.2f}%)")

    def create_ensemble_predictions(self, predictions):
        """
        Create ensemble predictions using weighted combination.

        For probability-based ensemble:
        - CatBoost outputs probabilities (0-1)
        - VAE outputs reconstruction errors (need to normalize)
        - FFNN outputs probabilities (0-1)

        We'll normalize VAE errors to 0-1 range and combine all with weights.
        """
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE PREDICTIONS")
        print("=" * 60)

        # Get scores from all models
        catboost_scores = predictions['catboost_probs']
        vae_scores = predictions['vae_errors']
        ffnn_scores = predictions['ffnn_probs']

        # Normalize VAE errors to 0-1 range (min-max normalization)
        vae_min = vae_scores.min()
        vae_max = vae_scores.max()
        vae_scores_norm = (vae_scores - vae_min) / (vae_max - vae_min + 1e-6)

        # Weighted combination
        ensemble_scores = (
            self.catboost_weight * catboost_scores +
            self.vae_weight * vae_scores_norm +
            self.ffnn_weight * ffnn_scores
        )

        print(f"\nEnsemble score statistics:")
        print(f"  Min: {ensemble_scores.min():.6f}")
        print(f"  Max: {ensemble_scores.max():.6f}")
        print(f"  Mean: {ensemble_scores.mean():.6f}")
        print(f"  Median: {np.median(ensemble_scores):.6f}")

        return ensemble_scores

    def optimize_ensemble_threshold(self, ensemble_scores):
        """Find optimal threshold for ensemble to minimize cost."""
        print("\n" + "=" * 60)
        print("OPTIMIZING ENSEMBLE THRESHOLD")
        print("=" * 60)

        # Search for optimal threshold
        thresholds = np.linspace(0.1, 0.9, 81)
        costs = []

        for threshold in thresholds:
            y_pred = (ensemble_scores >= threshold).astype(int)
            fp = ((self.y == 0) & (y_pred == 1)).sum()
            fn = ((self.y == 1) & (y_pred == 0)).sum()
            cost = fp * self.C_FP + fn * self.C_FN
            costs.append(cost)

        # Find minimum
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        min_cost = costs[optimal_idx]

        print(f"\nOptimal ensemble threshold: {optimal_threshold:.4f}")
        print(f"Minimum cost: ${min_cost:,.0f}")

        return optimal_threshold, min_cost

    def evaluate_ensemble(self, ensemble_scores, optimal_threshold):
        """Evaluate ensemble predictions with optimal threshold."""
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION")
        print("=" * 60)

        # Make predictions with optimal threshold
        ensemble_preds = (ensemble_scores >= optimal_threshold).astype(int)

        # Calculate metrics
        evaluator = FraudEvaluationMetrics(cost_fp=self.C_FP, cost_fn=self.C_FN)
        metrics = evaluator.calculate_metrics(
            self.y,
            ensemble_preds,
            y_scores=ensemble_scores
        )

        # Print results
        print("\n" + "=" * 60)
        print("=== ENSEMBLE RESULTS ON FULL DATASET ===")
        print("=" * 60)

        baseline_cost = self.y.sum() * self.C_FN
        savings = baseline_cost - metrics['total_cost']

        print("\nBusiness Cost Analysis:")
        print(f"  False Positives: {metrics['false_positives']:,} × ${self.C_FP} = ${metrics['false_positives'] * self.C_FP:,.0f}")
        print(f"  False Negatives: {metrics['false_negatives']:,} × ${self.C_FN} = ${metrics['false_negatives'] * self.C_FN:,.0f}")
        print(f"  Total Cost: ${metrics['total_cost']:,.0f}")
        print(f"\nBaseline Cost (no detection): ${baseline_cost:,.0f}")
        print(f"Net Savings: ${savings:,.0f} ({savings/baseline_cost*100:.2f}%)")

        print("\nClassification Metrics:")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        if metrics['pr_auc'] is not None:
            print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        else:
            print(f"  PR-AUC: N/A")

        print(f"\nPredictions:")
        print(f"  Total flagged as fraud: {ensemble_preds.sum():,}")
        print(f"  True Positives: {((self.y == 1) & (ensemble_preds == 1)).sum():,}")
        print(f"  True Negatives: {((self.y == 0) & (ensemble_preds == 0)).sum():,}")

        return {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'pr_auc': metrics['pr_auc'],
            'total_cost': metrics['total_cost'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'savings': savings,
            'baseline_cost': baseline_cost,
            'ensemble_scores': ensemble_scores,
            'ensemble_preds': ensemble_preds,
            'optimal_threshold': optimal_threshold
        }

    def run(self):
        """Execute complete ensemble pipeline."""
        print("\n" + "=" * 60)
        print("FRAUD DETECTION ENSEMBLE")
        print("=" * 60)

        # Load models
        self.load_models()

        # Load data
        self.load_data()

        # Get individual model predictions
        predictions = self.get_model_predictions()

        # Compute individual costs
        costs = self.compute_individual_costs(predictions)

        # Compute ensemble weights
        self.compute_ensemble_weights(costs)

        # Create ensemble predictions
        ensemble_scores = self.create_ensemble_predictions(predictions)

        # Optimize threshold
        optimal_threshold, min_cost = self.optimize_ensemble_threshold(ensemble_scores)

        # Evaluate ensemble
        results = self.evaluate_ensemble(ensemble_scores, optimal_threshold)

        # Add individual model metrics for comparison
        results['individual_models'] = {
            'catboost': costs['catboost'],
            'vae': costs['vae'],
            'ffnn': costs['ffnn']
        }
        results['weights'] = {
            'catboost': self.catboost_weight,
            'vae': self.vae_weight,
            'ffnn': self.ffnn_weight
        }

        print("\n" + "=" * 60)
        print("ENSEMBLE COMPLETE!")
        print("=" * 60)

        return results


def main():
    """
    Example usage of the ensemble module.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run fraud detection ensemble')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--catboost', type=str, default='models/catboost_best_tuned.cbm',
                       help='Path to CatBoost model')
    parser.add_argument('--vae', type=str, default='models/vae_best_tuned.pth',
                       help='Path to VAE model')
    parser.add_argument('--ffnn', type=str, default='models/ffnn_best_tuned.pth',
                       help='Path to FFNN model')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for PyTorch models')

    args = parser.parse_args()

    # Create ensemble
    ensemble = FraudEnsemble(
        catboost_model_path=args.catboost,
        vae_model_path=args.vae,
        ffnn_model_path=args.ffnn,
        data_path=args.data,
        device=args.device
    )

    # Run ensemble
    results = ensemble.run()

    # Save results
    import json
    results_path = 'results/ensemble_results.json'
    os.makedirs('results', exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_results = {
        'precision': convert_to_serializable(results['precision']),
        'recall': convert_to_serializable(results['recall']),
        'pr_auc': convert_to_serializable(results['pr_auc']),
        'total_cost': convert_to_serializable(results['total_cost']),
        'savings': convert_to_serializable(results['savings']),
        'optimal_threshold': convert_to_serializable(results['optimal_threshold']),
        'weights': {
            'catboost': convert_to_serializable(results['weights']['catboost']),
            'vae': convert_to_serializable(results['weights']['vae']),
            'ffnn': convert_to_serializable(results['weights']['ffnn'])
        }
    }

    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
