import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import joblib
import json
import os

def load_data(filepath="data/processed/features.csv"):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return pd.read_csv(filepath)

def train_and_evaluate(df):
    # Features and Target
    # X = df[['Home_Advantage', 'Home_Form_L5', 'Away_Form_L5', 'Position_Diff']]
    # Master Brief: Home_Advantage, Form_L5, Position_Diff
    feature_cols = ['Home_Advantage', 'Home_Form_L5', 'Away_Form_L5', 'Position_Diff']
    X = df[feature_cols]
    y = df['target'] # 0: Home, 1: Draw, 2: Away

    # Split data
    # Use last 20% for testing to simulate future prediction? Or random split?
    # Time-series data usually requires time-based split, but Master brief doesn't specify.
    # We'll use random split for now as standard classification.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 1. Baseline Model (Always Predict Home Win = 0)
    baseline_model = DummyClassifier(strategy="constant", constant=0)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    accuracy_base = accuracy_score(y_test, y_pred_base)
    print(f"\nBaseline Model Accuracy (Home Win Strategy): {accuracy_base:.4f}")

    # 2. AI Model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"AI Model (Random Forest) Accuracy: {accuracy_rf:.4f}")

    # Comparison
    print("\nResolution:")
    if accuracy_rf > accuracy_base:
        print("SUCCESS: AI Model outperforms Baseline.")
    else:
        print("WARNING: AI Model does not outperform Baseline.")
        
    print("\nClassification Report (AI Model):")
    print(classification_report(y_test, y_pred_rf, target_names=['Home Win', 'Draw', 'Away Win']))
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    print("AI Model saved to models/rf_model.pkl")
    
    # Process Feature Importance
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    feature_imp_df.to_csv("models/feature_importance.csv", index=False)
    print("Feature importance saved to models/feature_importance.csv")

    # Save metrics for Dashboard
    metrics = {
        "baseline_accuracy": accuracy_base,
        "ai_accuracy": accuracy_rf
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to models/metrics.json")

    return rf_model

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        train_and_evaluate(df)
