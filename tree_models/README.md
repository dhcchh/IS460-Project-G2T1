# Tree Models Package

This package contains modular implementations of gradient boosting models for credit card fraud detection.

## Structure

- `data_preprocessing.py`: Data loading and preprocessing utilities
- `model_definitions.py`: Model initialization and configuration
- `evaluation.py`: Custom fraud detection evaluation metrics with business cost analysis
- `visualizations.py`: Plotting and visualization functions
- `main.py`: Complete pipeline execution script
- `__init__.py`: Package initialization and exports

## Usage

### Run the complete pipeline:
```python
python main.py
```

### Use individual components:
```python
from tree_models import load_and_preprocess_data, get_models, FraudEvaluationMetrics

# Load data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

# Get models
models = get_models()

# Initialize custom evaluator
evaluator = FraudEvaluationMetrics(cost_fp=1, cost_fn=10)

# Train and evaluate a specific model
model = models['XGBoost']
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Get comprehensive metrics
metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
evaluator.print_metrics(y_test, y_pred, y_pred_proba)
evaluator.business_cost_analysis(y_test, y_pred)
```

## Models Included

1. **RandomForest**: Random Forest (baseline)
2. **XGBoost**: Extreme Gradient Boosting
3. **CatBoost**: Categorical Boosting
4. **LightGBM**: Light Gradient Boosting Machine

All models use consistent parameters for fair comparison:
- 100 estimators/trees
- Max depth of 6
- Random state of 42
- Parallel processing enabled

## Features

- **Custom Fraud Evaluation**: Specialized metrics for fraud detection including:
  - Precision, Recall, PR-AUC
  - Cost-sensitive evaluation with configurable FP/FN costs
  - Business cost analysis with financial impact
  - ROI calculations
- **Comprehensive Visualizations**: ROC curves, feature importance, confusion matrices
- **Training Time Comparison**: Performance benchmarking across models
- **Modular Design**: Easy to extend with new models or evaluation metrics


	•	Business Perspective: Cost matters most in fraud detection. CatBoost saves ~$4,000 more than XGBoost and ~$25,000 compared to LightGBM.
	•	Detection Quality: CatBoost has both the highest recall (captures more frauds, fewer false negatives) and the highest PR AUC, meaning more reliable fraud detection overall.
	•	Speed: XGBoost is slightly faster (1.04s vs 1.74s). But both are under 2 seconds → negligible difference in production.

⸻

