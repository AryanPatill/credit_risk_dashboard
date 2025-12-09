# model.py
"""
Train and save credit risk prediction model
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from config import MODEL_PARAMS, FEATURE_NAMES, RANDOM_STATE

def train_model():
    """
    Load data, train XGBoost model, and save it
    """
    print("📊 Loading data...")
    df = pd.read_csv('data/credit_data.csv')
    
    # Prepare features and target
    X = df[FEATURE_NAMES]
    y = df['is_default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🤖 Training XGBoost model...")
    model = xgb.XGBClassifier(
        **MODEL_PARAMS,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate
    print("\n📈 Model Performance:")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"✓ ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"✓ Accuracy: {model.score(X_test_scaled, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    print("\n💾 Saving model...")
    joblib.dump(model, 'models/credit_risk_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Model saved!")
    
    return model, scaler

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*50)
    print("🤖 TRAINING CREDIT RISK MODEL")
    print("="*50 + "\n")
    
    train_model()
    
    print("\n" + "="*50)
    print("✅ MODEL TRAINING COMPLETE")
    print("="*50)
    print("\nNext Step: Run 'streamlit run app.py'\n")

