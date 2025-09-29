"""
Model definitions for credit card fraud detection.
"""

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Common parameters for fair comparison
N_ESTIMATORS = 100
MAX_DEPTH = 6
RANDOM_STATE = 42
LEARNING_RATE = 0.1

def get_models():
    """
    Get dictionary of initialized models with consistent parameters for fair comparison.
    
    Returns:
        dict: Dictionary containing initialized models
    """
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1  # Use all available cores
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            objective='binary:logistic',
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=N_ESTIMATORS,
            depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            loss_function='Logloss',
            random_seed=RANDOM_STATE,
            verbose=False,
            train_dir=None,  # Prevent saving files
            allow_writing_files=False,  # Prevent saving any files
            thread_count=-1  # Use all available cores
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    }
    
    return models
