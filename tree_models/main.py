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
    
    print(f"\n{'='*20} {model_name} {'='*20}")
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics via evaluator
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    print(f"Training time: {train_time:.2f} seconds")
    evaluator.print_metrics(y_test, y_pred, y_pred_proba)
    evaluator.business_cost_analysis(y_test, y_pred)
    
    # Return results
    return {
        'model_name': model_name,
        'train_time': train_time,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'metrics': metrics,
        'feature_importance': getattr(model, 'feature_importances_', np.zeros(len(X_train.columns)))
    }

def main():
    
    # Initialize fraud evaluation metrics
    evaluator = FraudEvaluationMetrics(cost_fp=1, cost_fn=10)
    
    # 1. Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # 2. Get models
    models = get_models()
    
    # 3. Train and evaluate all models
    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate_model(model, name, X_train_scaled, X_test_scaled, y_train, y_test, evaluator)
    
    # 4. Generate visualizations
    output_dir = "results"
    plot_results(results, y_test, feature_names, output_dir)
    
if __name__ == "__main__":
    main()
