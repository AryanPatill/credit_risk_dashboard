# config.py
"""
Configuration file for Credit Risk Dashboard

This file centralizes all model parameters, feature names, and risk thresholds.
Modify these values to customize the model behavior.
"""

# Model Training Parameters
MODEL_PARAMS = {
    'n_estimators': 100,      # Number of boosting rounds
    'max_depth': 8,            # Maximum tree depth
    'learning_rate': 0.1,      # Learning rate (eta)
    'random_state': 42,        # Random seed for reproducibility
    'n_jobs': -1               # Use all available CPU cores
}

# Feature Names (must match training data columns)
FEATURE_NAMES = [
    'age',                     # Customer age in years
    'income',                  # Annual income in dollars
    'loan_amount',            # Requested loan amount in dollars
    'employment_years',       # Years of employment
    'credit_score',           # Credit score (300-850)
    'existing_debts',         # Total existing debts in dollars
    'monthly_payment_ratio',  # Monthly payment / income ratio
    'previous_defaults'       # Number of previous loan defaults
]

# Risk Category Thresholds
# Adjust these to change when a loan is classified as Low/Medium/High risk
RISK_LEVELS = {
    'Low': {
        'range': (0, 0.3),
        'color': '#27AE60',      # Green
        'description': 'Safe to approve'
    },
    'Medium': {
        'range': (0.3, 0.6),
        'color': '#F39C12',      # Orange
        'description': 'Review with caution'
    },
    'High': {
        'range': (0.6, 1.0),
        'color': '#E74C3C',      # Red
        'description': 'Recommend rejection'
    }
}

# Dataset Generation Parameters
DATASET_SIZE = 1000          # Number of synthetic samples to generate
RANDOM_STATE = 42            # Random seed for reproducibility
