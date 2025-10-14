"""
Credit Card Fraud Detection - Correlation Analysis Module

This module provides comprehensive correlation analysis tools for fraud detection datasets.
It identifies relationships between variables and highlights features most correlated with fraud.

INTENDED FOR JUPYTER NOTEBOOK IMPORTS:
This module is designed to be imported into Jupyter notebooks for interactive analysis.
Import specific functions as needed:

    from scripts.eda_helper.correlation_checker import check_variable_correlations, check_fraud_correlations
    
    # Analyze variable correlations
    var_corr = check_variable_correlations(df, top_n=15)
    
    # Analyze fraud correlations  
    fraud_corr = check_fraud_correlations(df, target_column='Class')
"""

from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Public API - Functions intended for import
__all__ = [
    'check_variable_correlations',
    'check_fraud_correlations',
    'run_full_correlation_analysis'
]


def _get_correlation_color(correlation: float) -> str:
    """
    Determine color coding for correlation strength.
    
    Args:
        correlation (float): Correlation coefficient value
        
    Returns:
        str: Color name for visualization
    """
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        return 'red'
    elif abs_corr > 0.5:
        return 'orange'
    else:
        return 'blue'


def _get_fraud_correlation_color(correlation: float) -> str:
    """
    Determine color coding for fraud correlation strength.
    
    Args:
        correlation (float): Correlation coefficient with fraud
        
    Returns:
        str: Color name for visualization
    """
    if correlation > 0.3:
        return 'darkred'
    elif correlation > 0.2:
        return 'red'
    elif correlation > 0.1:
        return 'orange'
    else:
        return 'blue'


def _extract_correlation_pairs(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique correlation pairs from correlation matrix.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        
    Returns:
        pd.DataFrame: DataFrame with correlation pairs and values
    """
    correlations = []
    
    for i, col_i in enumerate(corr_matrix.columns):
        for j, col_j in enumerate(corr_matrix.columns[i + 1:], start=i + 1):
            correlations.append({
                'var1': col_i,
                'var2': col_j,
                'correlation': corr_matrix.iloc[i, j]
            })
    
    correlations_df = pd.DataFrame(correlations)
    correlations_df['abs_correlation'] = correlations_df['correlation'].abs()
    
    return correlations_df.sort_values('abs_correlation', ascending=False)


def check_variable_correlations(
    df: pd.DataFrame, 
    top_n: int = 20,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Analyze and visualize correlations between all variable pairs in the dataset.
    
    Identifies the strongest relationships between variables and creates visualizations
    to highlight correlation patterns. Useful for feature selection and understanding
    data structure.
    
    JUPYTER NOTEBOOK USAGE:
        from scripts.eda_helper.correlation_checker import check_variable_correlations
        correlations = check_variable_correlations(df, top_n=15, show_plot=True)
    
    Args:
        df (pd.DataFrame): Input dataset with numerical variables
        top_n (int, optional): Number of top correlations to display. Defaults to 20.
        show_plot (bool, optional): Whether to display visualization. Defaults to True.
        
    Returns:
        pd.DataFrame: Sorted correlation pairs with columns:
            - var1: First variable name
            - var2: Second variable name  
            - correlation: Correlation coefficient
            - abs_correlation: Absolute correlation value
            
    Raises:
        ValueError: If dataframe is empty or has no numerical columns
        
    Example:
        >>> correlations = check_variable_correlations(credit_card_df, top_n=15)
        >>> print(f"Strongest correlation: {correlations.iloc[0]['correlation']:.3f}")
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numerical columns found in dataframe")
    
    print("=== VARIABLE CORRELATION ANALYSIS ===\n")
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Extract and sort correlation pairs
    correlations_df = _extract_correlation_pairs(corr_matrix)
    
    # Display results
    print(f"Top {top_n} highest correlations between variables:")
    display_cols = ['var1', 'var2', 'correlation']
    print(correlations_df[display_cols].head(top_n).to_string(index=False))
    
    # Create visualization if requested
    if show_plot:
        _plot_variable_correlations(correlations_df.head(top_n), top_n)
    
    return correlations_df


def _plot_variable_correlations(top_correlations: pd.DataFrame, top_n: int) -> None:
    """
    Create horizontal bar plot of top variable correlations.
    
    Args:
        top_correlations (pd.DataFrame): Top correlation pairs
        top_n (int): Number of correlations to plot
        
    Returns:
        None: Displays matplotlib plot
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare plot data
    labels = [
        f"{row['var1']} - {row['var2']}" 
        for _, row in top_correlations.iterrows()
    ]
    colors = [
        _get_correlation_color(corr) 
        for corr in top_correlations['correlation']
    ]
    
    # Create horizontal bar plot
    y_positions = list(enumerate(top_correlations.index))
    y_indices = [pos for pos, _ in y_positions]
    
    plt.barh(y_indices, top_correlations['correlation'], color=colors)
    plt.yticks(y_indices, labels, fontsize=8)
    plt.xlabel('Correlation Coefficient')
    plt.title(f'Top {top_n} Variable Correlations')
    plt.grid(axis='x', alpha=0.3)
    
    # Add reference lines
    reference_lines = [
        (0.7, 'red', 'Strong (>0.7)'),
        (-0.7, 'red', None),
        (0.5, 'orange', 'Moderate (>0.5)'),
        (-0.5, 'orange', None)
    ]
    
    for x_val, color, label in reference_lines:
        plt.axvline(
            x=x_val, color=color, linestyle='--', 
            alpha=0.5, label=label
        )
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def check_fraud_correlations(
    df: pd.DataFrame, 
    target_column: str = 'Class',
    show_plot: bool = True
) -> pd.Series:
    """
    Analyze and visualize which features are most correlated with fraud detection.
    
    Ranks all features by their absolute correlation with the target variable (fraud indicator)
    and creates comprehensive visualizations. Essential for feature importance analysis
    in fraud detection models.
    
    JUPYTER NOTEBOOK USAGE:
        from scripts.eda_helper.correlation_checker import check_fraud_correlations
        fraud_corrs = check_fraud_correlations(df, target_column='Class', show_plot=True)
    
    Args:
        df (pd.DataFrame): Input dataset containing features and target variable
        target_column (str, optional): Name of fraud indicator column. Defaults to 'Class'.
        show_plot (bool, optional): Whether to display visualization. Defaults to True.
        
    Returns:
        pd.Series: Features ranked by absolute correlation with fraud, indexed by feature name
        
    Raises:
        KeyError: If target_column not found in dataframe
        ValueError: If dataframe is empty or target column is not numerical
        
    Example:
        >>> fraud_corrs = check_fraud_correlations(credit_card_df, target_column='Class')
        >>> top_feature = fraud_corrs.index[0]
        >>> print(f"Most fraud-correlated feature: {top_feature} ({fraud_corrs.iloc[0]:.3f})")
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe")
    
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numerical")
    
    print(f"\n=== FEATURES MOST CORRELATED WITH {target_column.upper()} ===\n")
    
    # Calculate correlations with target variable
    correlations = df.corr()[target_column].drop(target_column)
    fraud_correlations = correlations.abs().sort_values(ascending=False)
    
    # Display ranked correlations
    print("Features ranked by absolute correlation with fraud:")
    for i, (feature, corr) in enumerate(fraud_correlations.items(), 1):
        print(f"{i:2d}. {feature:<8}: {corr:.4f}")
    
    # Create visualizations if requested
    if show_plot:
        _plot_fraud_correlations(df, fraud_correlations, target_column)
    
    # Show significant correlations
    _print_significant_correlations(fraud_correlations, target_column)
    
    return fraud_correlations


def _plot_fraud_correlations(
    df: pd.DataFrame, 
    fraud_correlations: pd.Series, 
    target_column: str
) -> None:
    """
    Create dual plot visualization for fraud correlations.
    
    Args:
        df (pd.DataFrame): Original dataframe
        fraud_correlations (pd.Series): Sorted fraud correlations
        target_column (str): Target column name
        
    Returns:
        None: Displays matplotlib subplots
    """
    top_features = fraud_correlations.head(10)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of top correlations
    colors = [_get_fraud_correlation_color(corr) for corr in top_features.values]
    y_positions = list(enumerate(top_features.index))
    y_indices = [pos for pos, _ in y_positions]
    
    ax1.barh(y_indices, top_features.values, color=colors)
    ax1.set_yticks(y_indices)
    ax1.set_yticklabels(top_features.index)
    ax1.set_xlabel('Absolute Correlation with Fraud')
    ax1.set_title('Top 10 Features Correlated with Fraud')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add threshold lines
    thresholds = [
        (0.3, 'darkred', 'Strong (>0.3)'),
        (0.2, 'red', 'Moderate (>0.2)'),
        (0.1, 'orange', 'Weak (>0.1)')
    ]
    
    for threshold, color, label in thresholds:
        ax1.axvline(
            x=threshold, color=color, linestyle='--', 
            alpha=0.5, label=label
        )
    ax1.legend()
    
    # Heatmap of correlations
    fraud_corr_matrix = pd.DataFrame(
        df.corr()[target_column][top_features.index]
    ).T
    
    sns.heatmap(
        fraud_corr_matrix, annot=True, cmap='RdBu_r', center=0,
        fmt='.3f', cbar_kws={'label': 'Correlation'}, ax=ax2
    )
    ax2.set_title('Correlation Heatmap: Top Features vs Fraud')
    
    plt.tight_layout()
    plt.show()


def _print_significant_correlations(
    fraud_correlations: pd.Series, 
    target_column: str,
    threshold: float = 0.1
) -> None:
    """
    Print features with correlation above significance threshold.
    
    Args:
        fraud_correlations (pd.Series): Fraud correlations
        target_column (str): Target column name
        threshold (float, optional): Significance threshold. Defaults to 0.1.
        
    Returns:
        None: Prints results to console
    """
    significant_features = fraud_correlations[fraud_correlations > threshold]
    
    print(f"\nFeatures with |correlation| > {threshold} with {target_column}:")
    for feature, corr in significant_features.items():
        print(f"  {feature}: {corr:.4f}")


def run_full_correlation_analysis(
    df: pd.DataFrame,
    target_column: str = 'Class',
    top_n: int = 20,
    show_plots: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run complete correlation analysis for fraud detection dataset.
    
    Performs both variable-to-variable correlation analysis and feature-to-fraud
    correlation analysis in a single function call. Ideal for comprehensive
    exploratory data analysis in Jupyter notebooks.
    
    JUPYTER NOTEBOOK USAGE:
        from scripts.eda_helper.correlation_checker import run_full_correlation_analysis
        var_corr, fraud_corr = run_full_correlation_analysis(df, show_plots=True)
    
    Args:
        df (pd.DataFrame): Input dataset containing features and target variable
        target_column (str, optional): Name of fraud indicator column. Defaults to 'Class'.
        top_n (int, optional): Number of top correlations to display. Defaults to 20.
        show_plots (bool, optional): Whether to display visualizations. Defaults to True.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - Variable correlation pairs (DataFrame)
            - Fraud correlation rankings (Series)
            
    Raises:
        ValueError: If dataframe is empty or has issues
        KeyError: If target_column not found in dataframe
        
    Example:
        >>> var_corr, fraud_corr = run_full_correlation_analysis(credit_card_df)
        >>> print(f"Top fraud predictor: {fraud_corr.index[0]} ({fraud_corr.iloc[0]:.3f})")
        >>> print(f"Strongest variable correlation: {var_corr.iloc[0]['correlation']:.3f}")
    """
    print(f"=== COMPREHENSIVE CORRELATION ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    
    if target_column in df.columns:
        fraud_count = df[target_column].sum()
        fraud_rate = df[target_column].mean() * 100
        print(f"Fraud cases: {fraud_count} ({fraud_rate:.2f}%)")
    
    print("-" * 60)
    
    # Perform correlation analyses
    print("Starting variable correlation analysis...")
    var_correlations = check_variable_correlations(df, top_n=top_n, show_plot=show_plots)
    
    print("\nStarting fraud correlation analysis...")
    fraud_correlations = check_fraud_correlations(df, target_column=target_column, show_plot=show_plots)
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"✓ Analyzed {len(var_correlations)} variable correlation pairs")
    print(f"✓ Ranked {len(fraud_correlations)} features by fraud correlation")
    print(f"✓ Strongest variable correlation: {var_correlations.iloc[0]['correlation']:.4f}")
    print(f"✓ Top fraud predictor: {fraud_correlations.index[0]} ({fraud_correlations.iloc[0]:.4f})")
    
    return var_correlations, fraud_correlations


def main() -> None:
    """
    Standalone execution function for command-line usage.
    
    This function is kept for backward compatibility and command-line execution.
    For Jupyter notebook usage, prefer run_full_correlation_analysis().
    """
    try:
        # Load dataset
        df = pd.read_csv('../../data/raw/creditcard.csv')
        
        # Run full analysis
        var_correlations, fraud_correlations = run_full_correlation_analysis(df)
        
    except FileNotFoundError:
        print("Error: Could not find creditcard.csv in ../../data/raw/")
        print("Please ensure the dataset is in the correct location.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()