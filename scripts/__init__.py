"""
EDA Helper Package

Utility functions for exploratory data analysis in fraud detection projects.
"""

from .correlation_checker import (
    check_variable_correlations, 
    check_fraud_correlations,
    run_full_correlation_analysis
)

__all__ = [
    'check_variable_correlations', 
    'check_fraud_correlations',
    'run_full_correlation_analysis'
]