from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, Optional, Union, List

class FraudEvaluationMetrics:
    """
    Evaluation metrics for credit card fraud detection.
    Handles imbalanced dataset evaluation with cost-sensitive total cost calculation.
    """
 
    def __init__(self, cost_fp: int = 110, cost_fn: int = 540) -> None:
        """
        Initialize with false positive and false negative costs.
        
        Args:
            cost_fp (int): Cost of false positive (flagging normal as fraud)
            cost_fn (int): Cost of false negative (missing actual fraud)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
         
    def calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], y_scores: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Union[float, int, None]]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true (Union[pd.Series, np.ndarray]): True labels (0=normal, 1=fraud)
            y_pred (Union[pd.Series, np.ndarray]): Predicted labels (0=normal, 1=fraud)  
            y_scores (Optional[Union[pd.Series, np.ndarray]]): Prediction scores/probabilities for PR AUC
            
        Returns:
            Dict[str, Union[float, int, None]]: Dictionary containing all metrics
        """
        # Convert to numpy arrays for consistent processing
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_scores is not None:
            y_scores = np.array(y_scores)
        # Basic counts
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # PR AUC (requires scores)
        pr_auc = None
        if y_scores is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall_curve, precision_curve)
        
        # Total cost
        total_cost = self.cost_fp * fp + self.cost_fn * fn
        
        return {
            'precision': precision,
            'recall': recall, 
            'pr_auc': pr_auc,
            'total_cost': total_cost,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def plot_pr_curve(self, y_true: Union[pd.Series, np.ndarray], y_scores: Union[pd.Series, np.ndarray], title: str = "Precision-Recall Curve", figsize: tuple = (8, 6)) -> float:
        """
        Plot the Precision-Recall curve for fraud detection model.

        Args:
            y_true (Union[pd.Series, np.ndarray]): True labels (0=normal, 1=fraud)
            y_scores (Union[pd.Series, np.ndarray]): Prediction scores/probabilities
            title (str): Title for the plot
            figsize (tuple): Figure size (width, height)

        Returns:
            float: PR AUC score
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        baseline = np.sum(y_true) / len(y_true)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.axhline(y=baseline, color='red', linestyle='--', linewidth=1, label=f'Baseline (Random) = {baseline:.4f}')

        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.text(0.6, 0.2, f'PR AUC: {pr_auc:.4f}\nBaseline: {baseline:.4f}\nImprovement: {pr_auc/baseline:.2f}x',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()
        plt.show()

        return pr_auc
    
    
    def business_cost_analysis(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], logistic_baseline_cost: Optional[int] = None) -> Dict[str, Union[float, int]]:
        """
        Business cost analysis focusing on Total_Cost calculation.

        Args:
            y_true (Union[pd.Series, np.ndarray]): True labels
            y_pred (Union[pd.Series, np.ndarray]): Predicted labels
            logistic_baseline_cost (Optional[int]): Logistic regression baseline cost for comparison (optional)

        Returns:
            Dict[str, Union[float, int]]: Business cost breakdown
        """
        # Convert to numpy arrays for consistent processing
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # Calculate total cost: Total_Cost = sum(fp * fp_cost) + sum(fn * fn_cost)
        fp_cost_total = fp * self.cost_fp
        fn_cost_total = fn * self.cost_fn
        total_cost = fp_cost_total + fn_cost_total

        # Calculate baseline cost (no fraud detection - all frauds would be missed)
        total_frauds = np.sum(y_true == 1)
        baseline_cost = total_frauds * self.cost_fn
        cost_savings_from_baseline = baseline_cost - total_cost

        print("=== BUSINESS COST ANALYSIS ===")
        print(f"False Positive Cost (FP × {self.cost_fp}): {fp_cost_total}")
        print(f"False Negative Cost (FN × {self.cost_fn}): {fn_cost_total}")
        print(f"Total Cost: ${total_cost}")
        print(f"Baseline Cost (No Detection): ${baseline_cost}")
        print(f"Cost Savings vs No Detection: ${cost_savings_from_baseline} ({(cost_savings_from_baseline/baseline_cost)*100:.2f}%)")

        # Only print logistic regression comparison if provided
        if logistic_baseline_cost is not None:
            cost_savings_from_logistic = logistic_baseline_cost - total_cost
            print(f"Logistic Regression Cost: ${logistic_baseline_cost}")
            print(f"Cost Savings vs Logistic Regression: ${cost_savings_from_logistic} ({(cost_savings_from_logistic/logistic_baseline_cost)*100:.2f}%)")

        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")

        return {
            'fp_cost_total': fp_cost_total,
            'fn_cost_total': fn_cost_total,
            'total_cost': total_cost,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn
        }

    def save_results(
        self,
        results_dict: Dict,
        save_path: str = "results/model_results.json"
    ) -> Dict:
        """
        Save business cost analysis results to JSON file.

        Args:
            results_dict: Results dictionary from business_cost_analysis() method
            save_path: Path to save JSON file

        Returns:
            Dict: The results dictionary that was saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        results_to_save = {}
        for key, value in results_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                results_to_save[key] = value.item()
            elif isinstance(value, np.integer):
                results_to_save[key] = int(value)
            elif isinstance(value, np.floating):
                results_to_save[key] = float(value)
            else:
                results_to_save[key] = value

        # Add timestamp and cost parameters
        results_to_save['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results_to_save['cost_fp'] = self.cost_fp
        results_to_save['cost_fn'] = self.cost_fn

        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\nResults saved to: {save_path}")
        print(f"Total Cost: ${results_to_save['total_cost']:,}")

        return results_to_save