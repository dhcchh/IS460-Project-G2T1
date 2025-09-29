from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

class FraudEvaluationMetrics:
    """
    Evaluation metrics for credit card fraud detection.
    Handles imbalanced dataset evaluation with cost-sensitive total cost calculation.
    """
  
    def __init__(self, cost_fp: int = 1, cost_fn: int = 10) -> None:
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
    
    def print_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], y_scores: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Union[float, int, None]]:
        """
        Print formatted metrics results.
        
        Args:
            y_true (Union[pd.Series, np.ndarray]): True labels
            y_pred (Union[pd.Series, np.ndarray]): Predicted labels  
            y_scores (Optional[Union[pd.Series, np.ndarray]]): Prediction scores/probabilities for PR AUC
            
        Returns:
            Dict[str, Union[float, int, None]]: Dictionary containing all metrics
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_scores)
        
        print("=== FRAUD DETECTION METRICS ===")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        if metrics['pr_auc'] is not None:
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Total Cost: {metrics['total_cost']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        return metrics
    
    def business_cost_analysis(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], avg_transaction_amount: float = 100.0) -> Dict[str, Union[float, int]]:
        """
        Detailed business cost analysis with financial impact.
        
        Args:
            y_true (Union[pd.Series, np.ndarray]): True labels
            y_pred (Union[pd.Series, np.ndarray]): Predicted labels
            avg_transaction_amount (float): Average transaction amount for cost estimation
            
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
        
        # Business costs
        investigation_cost = fp * self.cost_fp  # Cost to investigate false alarms
        fraud_loss = fn * self.cost_fn * avg_transaction_amount  # Actual fraud losses
        total_business_cost = investigation_cost + fraud_loss
        
        # Prevented fraud value
        prevented_fraud = tp * avg_transaction_amount
        
        # Cost per transaction
        total_transactions = len(y_true)
        cost_per_transaction = total_business_cost / total_transactions
        
        print("=== BUSINESS COST ANALYSIS ===")
        print(f"Investigation costs (FP): ${investigation_cost:,.2f}")
        print(f"Fraud losses (FN): ${fraud_loss:,.2f}")
        print(f"Total business cost: ${total_business_cost:,.2f}")
        print(f"Prevented fraud value: ${prevented_fraud:,.2f}")
        print(f"Cost per transaction: ${cost_per_transaction:.4f}")
        print(f"ROI: {(prevented_fraud - total_business_cost)/total_business_cost*100:.2f}%" if total_business_cost > 0 else "N/A")
        
        return {
            'investigation_cost': investigation_cost,
            'fraud_loss': fraud_loss,
            'total_business_cost': total_business_cost,
            'prevented_fraud': prevented_fraud,
            'cost_per_transaction': cost_per_transaction,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn
        }