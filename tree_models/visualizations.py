"""
Visualization utilities for model comparison.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def _ensure_dir(output_dir: str) -> None:
    """Create the output directory if it does not exist."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def plot_roc_curves(results, y_test, output_dir: str = "results", figsize=(10, 8)):
    """
    Plot ROC curves for all models.
    
    Args:
        results: Dictionary containing model results
        output_dir: Directory to save the figure
        y_test: Test labels
        figsize: Figure size tuple
    """
    from sklearn.metrics import roc_curve, auc
    
    _ensure_dir(output_dir)
    plt.figure(figsize=figsize)
    for name in results:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def plot_feature_importance(results, feature_names, output_dir: str = "results", figsize=(20, 12)):
    """
    Plot feature importance for all models.
    
    Args:
        results: Dictionary containing model results
        feature_names: List of feature names
        output_dir: Directory to save the figure
        figsize: Figure size tuple
    """
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Feature Importance Comparison')
    
    for idx, (name, result) in enumerate(results.items()):
        importance = result['feature_importance']
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        # Convert feature_names to numpy array for proper indexing
        feature_names_array = np.array(feature_names)
        
        # Flatten axes for 2x2 grid
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ax.barh(pos, importance[sorted_idx])
        ax.set_yticks(pos)
        ax.set_yticklabels(feature_names_array[sorted_idx])
        ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_confusion_matrices(results, y_test, output_dir: str = "results", figsize=(20, 10)):
    """
    Plot confusion matrices for all models.
    
    Args:
        results: Dictionary containing model results
        y_test: Test labels
        output_dir: Directory to save the figure
        figsize: Figure size tuple
    """
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Confusion Matrices Comparison')
    
    for idx, (name, result) in enumerate(results.items()):
        conf_matrix = confusion_matrix(y_test, result['predictions'])
        
        # Flatten axes for 2x2 grid
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

def plot_training_times(results, output_dir: str = "results", figsize=(8, 5)):
    """
    Plot training time comparison.
    
    Args:
        results: Dictionary containing model results
        output_dir: Directory to save the figure
        figsize: Figure size tuple
    """
    _ensure_dir(output_dir)
    plt.figure(figsize=figsize)
    times = [results[name]['train_time'] for name in results]
    plt.bar(results.keys(), times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_times.png'))
    plt.close()


def plot_results(results, y_test, feature_names, output_dir: str = "results") -> None:
    """
    Generate and save all result visualizations to disk.

    Args:
        results: Dictionary of model results from the pipeline
        y_test: Ground-truth test labels
        feature_names: Sequence of feature names corresponding to model inputs
        output_dir: Directory where plots will be saved
    """
    _ensure_dir(output_dir)
    plot_roc_curves(results, y_test, output_dir)
    plot_feature_importance(results, feature_names, output_dir)
    plot_confusion_matrices(results, y_test, output_dir)
    plot_training_times(results, output_dir)
