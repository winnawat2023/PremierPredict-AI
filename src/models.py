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

    # 1. Baseline Model 1 (Home Win Strategy)
    # Always Predict Home Win (0)
    baseline1_model = DummyClassifier(strategy="constant", constant=0)
    baseline1_model.fit(X_train, y_train)
    y_pred_base1 = baseline1_model.predict(X_test)
    accuracy_base1 = accuracy_score(y_test, y_pred_base1)
    print(f"\nBaseline 1 (Home Win Strategy) Accuracy: {accuracy_base1:.4f}")

    # 2. Baseline Model 2 (Higher Rank Strategy)
    # Predict based on Position_Diff (Home Rank - Away Rank)
    # if < 0 (Home better), predict 0 (Home Win)
    # if > 0 (Away better), predict 2 (Away Win)
    # if == 0, predict 1 (Draw) - or simply Home Win
    y_pred_base2 = []
    for diff in X_test['Position_Diff']:
        if diff < 0:
            y_pred_base2.append(0) # Home Win
        elif diff > 0:
            y_pred_base2.append(2) # Away Win
        else:
            y_pred_base2.append(0) # Assume Home Win if rank equal
    
    accuracy_base2 = accuracy_score(y_test, y_pred_base2)
    print(f"Baseline 2 (Higher Rank Strategy) Accuracy: {accuracy_base2:.4f}")

    # 3. Baseline Model 3 (Probabilistic)
    # Random prediction based on training set distribution
    baseline3_model = DummyClassifier(strategy="stratified", random_state=42)
    baseline3_model.fit(X_train, y_train)
    y_pred_base3 = baseline3_model.predict(X_test)
    accuracy_base3 = accuracy_score(y_test, y_pred_base3)
    print(f"Baseline 3 (Probabilistic) Accuracy: {accuracy_base3:.4f}")

    # 2. AI Model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"AI Model (Random Forest) Accuracy: {accuracy_rf:.4f}")

    # Comparison
    print("\nResolution:")
    if accuracy_rf > accuracy_base1:
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
        "baseline1_accuracy": accuracy_base1,
        "baseline2_accuracy": accuracy_base2,
        "baseline3_accuracy": accuracy_base3,
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
