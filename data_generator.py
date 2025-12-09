# data_generator.py
"""
Generate synthetic credit dataset for model training
"""
import pandas as pd
import numpy as np
from config import DATASET_SIZE, RANDOM_STATE, FEATURE_NAMES

np.random.seed(RANDOM_STATE)

def generate_credit_data(n_samples=DATASET_SIZE):
    """
    Generate synthetic credit risk dataset
    """
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'loan_amount': np.random.randint(5000, 500000, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'existing_debts': np.random.randint(0, 100000, n_samples),
        'monthly_payment_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'previous_defaults': np.random.randint(0, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (default risk) based on logical rules
    # Higher credit score, income, lower debt = lower risk
    risk_score = (
        (850 - df['credit_score']) / 550 * 0.3 +
        (df['monthly_payment_ratio']) * 0.3 +
        (df['previous_defaults'] / 5) * 0.2 +
        (df['existing_debts'] / df['income'].max()) * 0.2
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.05, n_samples)
    df['default_risk'] = np.clip(risk_score, 0, 1)
    
    # Binary classification: 1 if risk > 0.5, else 0
    df['is_default'] = (df['default_risk'] > 0.5).astype(int)
    
    return df

# At the end of data_generator.py, replace the main section with:

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    df = generate_credit_data()
    df.to_csv('data/credit_data.csv', index=False)
    
    print("\n" + "="*50)
    print("✅ DATASET GENERATED SUCCESSFULLY")
    print("="*50)
    print(f"📊 Total Records: {len(df)}")
    print(f"📊 Default Rate: {df['is_default'].mean():.2%}")
    print(f"📊 File Location: data/credit_data.csv")
    print("="*50 + "\n")
    print("Next Step: Run 'python model.py' to train the model\n")
