# Credit Card Fraud Dataset 

Create your own `data` folder with `processed` and `raw` subfolders 

Run `EDA_named_features_analysis_FE` first , then run `EDA_statistical_analysis.ipynb`




.
├── notebooks
│   ├── catboost_model_test
│   │   ├── catboost_baseline_training.ipynb
│   │   └── catboost_hyperparameter_tuning.ipynb
│   └── vae_model_test
│       ├── vae_baseline_training.ipynb
│       └── vae_hyperparameter_tuning.ipynb
├── src
│   ├── __init__.py
│   ├── catboost_models
│   │   ├── __init__.py
│   │   ├── catboost_base.py
│   │   ├── catboost_trainer.py
│   │   └── catboost_tuner.py
│   └── vae_models
│       ├── __init__.py
│       ├── vae_base.py
│       ├── vae_trainer.py
│       └── vae_tuner.py