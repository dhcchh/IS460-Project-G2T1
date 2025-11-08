"""
Feature Engineering for Credit Card Fraud Detection

This module creates engineered features from the raw creditcard dataset.
Features are based on EDA analysis from notebooks/eda/EDA_named_features_analysis_FE.ipynb

Engineered Features:
- Hour: Hour of day (0-23) extracted from Time
- Hour_sin: Cyclical encoding of hour using sine transformation
- Day: Day number extracted from Time
- log_Amount: Log-transformed Amount to handle skewness
- Time_bin: Time of day category (0=Night, 1=Morning, 2=Afternoon, 3=Evening)
- Amount_bin: Amount category (0=Low, 1=Medium, 2=High) based on quantiles
"""

import os
import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw creditcard data.

    Args:
        df: DataFrame with raw creditcard data containing 'Time' and 'Amount' columns

    Returns:
        DataFrame with engineered features added
    """
    # Make a copy to avoid modifying original
    df_fe = df.copy()

    # 1. Hour: Extract hour from Time (seconds elapsed)
    # Time is in seconds, so divide by 3600 to get hours, then mod 24 for hour of day
    df_fe['Hour'] = (df_fe['Time'] // 3600) % 24

    # 2. Hour_sin: Cyclical encoding of hour using sine transformation
    # Hours are cyclical (0-23 wraps around), so use sine to capture this
    df_fe['Hour_sin'] = np.sin(2 * np.pi * df_fe['Hour'] / 24)

    # 3. Day: Extract day number from Time
    df_fe['Day'] = df_fe['Time'] // (3600 * 24)

    # 4. log_Amount: Log transformation of Amount to handle skewness
    # Add small epsilon to avoid log(0) issues
    df_fe['log_Amount'] = np.log(df_fe['Amount'] + 1e-6)

    # 5. Time_bin: Categorize time of day
    # 0-6: Night, 6-12: Morning, 12-18: Afternoon, 18-24: Evening
    df_fe['Time_bin'] = pd.cut(
        df_fe['Hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )

    # 6. Amount_bin: Categorize amount by quantiles
    # Split into 3 bins: Low, Medium, High based on quantiles
    df_fe['Amount_bin'] = pd.qcut(
        df_fe['Amount'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # Convert categorical bins to numeric codes
    # Night=0, Morning=1, Afternoon=2, Evening=3
    df_fe['Time_bin'] = df_fe['Time_bin'].cat.codes

    # Low=0, Medium=1, High=2
    df_fe['Amount_bin'] = df_fe['Amount_bin'].cat.codes

    return df_fe


def create_feature_engineered_dataset(
    input_path: str = 'data/raw/creditcard.csv',
    output_path: str = 'data/processed/creditcard_fe.csv',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load raw creditcard data, apply feature engineering, and save to processed folder.

    Args:
        input_path: Path to raw creditcard.csv
        output_path: Path to save feature-engineered CSV
        verbose: Print progress messages

    Returns:
        DataFrame with engineered features
    """
    if verbose:
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        print(f"\nLoading raw data from: {input_path}")

    # Load raw data
    df = pd.read_csv(input_path)

    if verbose:
        print(f"Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

    # Apply feature engineering
    if verbose:
        print("\nApplying feature engineering...")
        print("  - Extracting Hour from Time")
        print("  - Creating cyclical Hour_sin feature")
        print("  - Extracting Day from Time")
        print("  - Log-transforming Amount")
        print("  - Creating Time_bin (Night/Morning/Afternoon/Evening)")
        print("  - Creating Amount_bin (Low/Medium/High)")

    df_fe = engineer_features(df)

    if verbose:
        print(f"\nFeature engineering complete!")
        print(f"  New columns added: Hour, Hour_sin, Day, log_Amount, Time_bin, Amount_bin")
        print(f"  Total columns: {len(df_fe.columns)}")
        print(f"  Final shape: {df_fe.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_fe.to_csv(output_path, index=False)

    if verbose:
        print(f"\nFeature-engineered dataset saved to: {output_path}")
        print("=" * 60)

    return df_fe


def get_logreg_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the feature subset used by the logistic regression baseline model.

    This drops features that were found to be less important:
    - V13, V15, V19, V20, V22, V23, V24, V25, V26
    - Time, Amount, Hour, log_Amount

    Keeps: V1-V12, V14, V16-V18, V21, V27-V28, Hour_sin, Day, Time_bin, Amount_bin
    Total: 22 features (used by logreg baseline)

    Args:
        df: DataFrame with all features (including engineered ones)

    Returns:
        DataFrame with baseline feature subset and Class column
    """
    features_to_drop = [
        'V13', 'V15', 'V19', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26',
        'Time', 'Amount', 'Hour', 'log_Amount'
    ]

    # Drop features but keep Class if it exists
    if 'Class' in df.columns:
        df_baseline = df.drop(columns=features_to_drop)
    else:
        df_baseline = df.drop(columns=features_to_drop)

    return df_baseline


def print_feature_summary(df: pd.DataFrame):
    """
    Print a summary of engineered features and their statistics.

    Args:
        df: DataFrame with engineered features
    """
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    print("\n1. Original Features:")
    print(f"   - Time (seconds elapsed): min={df['Time'].min():.0f}, max={df['Time'].max():.0f}")
    print(f"   - Amount: min=${df['Amount'].min():.2f}, max=${df['Amount'].max():.2f}")
    print(f"   - V1-V28: PCA-transformed features")

    print("\n2. Engineered Time Features:")
    print(f"   - Hour: Range 0-23, unique values: {df['Hour'].nunique()}")
    print(f"   - Hour_sin: Range [{df['Hour_sin'].min():.3f}, {df['Hour_sin'].max():.3f}]")
    print(f"   - Day: Range 0-{int(df['Day'].max())}, unique values: {df['Day'].nunique()}")
    print(f"   - Time_bin: {df['Time_bin'].value_counts().to_dict()} (0=Night, 1=Morning, 2=Afternoon, 3=Evening)")

    print("\n3. Engineered Amount Features:")
    print(f"   - log_Amount: Range [{df['log_Amount'].min():.3f}, {df['log_Amount'].max():.3f}]")
    print(f"   - Amount_bin: {df['Amount_bin'].value_counts().to_dict()} (0=Low, 1=Medium, 2=High)")

    if 'Class' in df.columns:
        print("\n4. Target Distribution:")
        print(f"   - Normal: {(df['Class'] == 0).sum()} ({(df['Class'] == 0).sum() / len(df) * 100:.2f}%)")
        print(f"   - Fraud: {(df['Class'] == 1).sum()} ({(df['Class'] == 1).sum() / len(df) * 100:.2f}%)")

    print("\n5. Feature Importance Insights (from EDA):")
    print("   - Night transactions have higher fraud rate (0.52% vs ~0.14% other times)")
    print("   - Low amount transactions have slightly higher fraud rate")
    print("   - Hour_sin captures cyclical nature of time")

    print("=" * 60)


def main():
    """
    Main function to run feature engineering pipeline.
    Usage: python src/feature_engineering.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering for credit card fraud detection')
    parser.add_argument('--input', type=str, default='data/raw/creditcard.csv',
                       help='Path to raw creditcard.csv')
    parser.add_argument('--output', type=str, default='data/processed/creditcard_fe.csv',
                       help='Path to save feature-engineered CSV')
    parser.add_argument('--summary', action='store_true',
                       help='Print feature summary after engineering')
    parser.add_argument('--baseline-only', action='store_true',
                       help='Also save logistic regression baseline feature subset')

    args = parser.parse_args()

    # Run feature engineering
    df_fe = create_feature_engineered_dataset(
        input_path=args.input,
        output_path=args.output,
        verbose=True
    )

    # Print summary if requested
    if args.summary:
        print_feature_summary(df_fe)

    # Save baseline subset if requested
    if args.baseline_only:
        baseline_output = args.output.replace('.csv', '_baseline.csv')
        df_baseline = get_logreg_baseline_features(df_fe)
        df_baseline.to_csv(baseline_output, index=False)
        print(f"\nLogistic regression baseline features saved to: {baseline_output}")
        print(f"  Features: {len(df_baseline.columns) - 1} (excluding Class)")

    print("\n✓ Feature engineering complete!")


if __name__ == '__main__':
    main()
