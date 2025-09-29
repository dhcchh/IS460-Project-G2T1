"""
Main script to run the complete model comparison pipeline.
"""

import numpy as np
import time
from data_preprocessing import load_and_preprocess_data
from evaluation import FraudEvaluationMetrics
from model_definitions import get_models
from visualizations import plot_results

# Set random seed for reproducibility
np.random.seed(42)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, evaluator):
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics via evaluator
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Return results
    return {
        'model_name': model_name,
        'train_time': train_time,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics,
        'feature_importance': getattr(model, 'feature_importances_', np.zeros(len(X_train.columns)))
    }

def print_comparison_summary(results, y_test, evaluator):
    """Print side-by-side comparison of all models using evaluator metrics and business cost analysis."""
    
    print("\n" + "="*120)
    print("MODEL COMPARISON SUMMARY - METRICS")
    print("="*120)
    
    # Header - only metrics provided by evaluator
    print(f"{'Model':<15} {'Train Time':<12} {'Precision':<12} {'Recall':<12} {'PR-AUC':<12} {'Total Cost':<12} {'FP':<8} {'FN':<8}")
    print("-" * 120)
    
    # Results for each model
    for name, result in results.items():
        metrics = result['metrics']
        train_time = result['train_time']
        
        # Format PR-AUC (can be None)
        pr_auc_str = f"{metrics['pr_auc']:.4f}" if metrics['pr_auc'] is not None else "N/A"
        
        print(f"{name:<15} {train_time:<12.2f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{pr_auc_str:<12} {metrics['total_cost']:<12} {metrics['false_positives']:<8} {metrics['false_negatives']:<8}")
    
    print("\n" + "="*120)
    print("MODEL COMPARISON SUMMARY - BUSINESS COSTS")
    print("="*120)
    
    # Header for business cost analysis
    print(f"{'Model':<15} {'Investigation':<15} {'Fraud Loss':<15} {'Total Business':<15} {'Prevented':<15} {'Cost/Trans':<12} {'ROI':>12}")
    print("-" * 120)
    
    # Collect business cost data for each model (suppress individual printing)
    business_data = {}
    for name, result in results.items():
        # Get business cost analysis without printing
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture the output to suppress printing
        f = io.StringIO()
        with redirect_stdout(f):
            cost_analysis = evaluator.business_cost_analysis(y_test, result['predictions'])
        
        # Calculate ROI manually since it's not returned by the method
        roi = ((cost_analysis['prevented_fraud'] - cost_analysis['total_business_cost']) / cost_analysis['total_business_cost'] * 100) if cost_analysis['total_business_cost'] > 0 else 0
        cost_analysis['roi'] = roi
        
        business_data[name] = cost_analysis
        
        print(f"{name:<15} ${cost_analysis['investigation_cost']:<14,.0f} ${cost_analysis['fraud_loss']:<14,.0f} "
              f"${cost_analysis['total_business_cost']:<14,.0f} ${cost_analysis['prevented_fraud']:<14,.0f} "
              f"${cost_analysis['cost_per_transaction']:<11.4f} {roi:>11.2f}%")
    
    print("\n" + "="*120)
    print("BEST MODEL RECOMMENDATIONS")
    print("="*120)
    
    # Find best models by different criteria (metrics)
    best_precision = max(results.items(), key=lambda x: x[1]['metrics']['precision'])
    best_recall = max(results.items(), key=lambda x: x[1]['metrics']['recall'])
    best_pr_auc = max(results.items(), key=lambda x: x[1]['metrics']['pr_auc'] if x[1]['metrics']['pr_auc'] is not None else 0)
    lowest_cost = min(results.items(), key=lambda x: x[1]['metrics']['total_cost'])
    fastest = min(results.items(), key=lambda x: x[1]['train_time'])
    
    print("METRICS-BASED RECOMMENDATIONS:")
    print(f"Best Precision: {best_precision[0]} ({best_precision[1]['metrics']['precision']:.4f})")
    print(f"Best Recall: {best_recall[0]} ({best_recall[1]['metrics']['recall']:.4f})")
    if best_pr_auc[1]['metrics']['pr_auc'] is not None:
        print(f"Best PR-AUC: {best_pr_auc[0]} ({best_pr_auc[1]['metrics']['pr_auc']:.4f})")
    print(f"Lowest Cost: {lowest_cost[0]} ({lowest_cost[1]['metrics']['total_cost']})")
    print(f"Fastest Training: {fastest[0]} ({fastest[1]['train_time']:.2f}s)")
    
    # Find best models by business criteria
    best_roi = max(business_data.items(), key=lambda x: x[1]['roi'])
    lowest_business_cost = min(business_data.items(), key=lambda x: x[1]['total_business_cost'])
    highest_prevented = max(business_data.items(), key=lambda x: x[1]['prevented_fraud'])
    
    print("\nBUSINESS-BASED RECOMMENDATIONS:")
    print(f"Best ROI: {best_roi[0]} ({best_roi[1]['roi']:.2f}%)")
    print(f"Lowest Business Cost: {lowest_business_cost[0]} (${lowest_business_cost[1]['total_business_cost']:,.0f})")
    print(f"Highest Prevented Fraud: {highest_prevented[0]} (${highest_prevented[1]['prevented_fraud']:,.0f})")

def main():
    
    # Initialize fraud evaluation metrics
    evaluator = FraudEvaluationMetrics(cost_fp=1, cost_fn=10)
    
    # 1. Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # 2. Get models
    models = get_models()
    
    # 3. Train and evaluate all models
    print("Training and evaluating models...")
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        results[name] = train_and_evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test, evaluator)
    
    # 4. Print comparison summary
    print_comparison_summary(results, y_test, evaluator)
    
    # 5. Generate visualizations
    print("\nGenerating visualizations...")
    output_dir = "results"
    plot_results(results, y_test, feature_names, output_dir)
    
if __name__ == "__main__":
    main()
