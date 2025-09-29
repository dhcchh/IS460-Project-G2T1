"""
Data preprocessing module for credit card fraud detection models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path='../data/processed/creditcard_fe.csv'):
    """
    Load and preprocess the credit card fraud detection dataset.
    
    Args:
        data_path (str): Path to the dataset
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    # Display basic information about the dataset
    print("Dataset Shape:", data.shape)
    print("\nClass Distribution:")
    print(data['Class'].value_counts(normalize=True))
    
    # Prepare features and target
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
