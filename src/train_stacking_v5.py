
import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from src.utils import normalize_team_name
from src.constants import BASE_FEATURES, HYBRID_FEATURES

def prepare_data():
    print("Preparing data for Stacking Ensemble V5...")
    # 1. Load data
    features_df = pd.read_csv('data/processed/features.csv')
    crowd_df = pd.read_csv('data/raw/crowd_predictions_2024.csv')
    fpl_df = pd.read_csv('data/raw/fpl_baselines_2024.csv')

    # 2. Prepare FPL scores
    fpl_df['name_norm'] = fpl_df['name'].apply(normalize_team_name)
    fpl_scores = dict(zip(fpl_df['name_norm'], fpl_df['fan_ownership_score']))

    # 3. Augment data with human-signal features
    df = features_df.copy().dropna(subset=BASE_FEATURES)
    df['fpl_home'] = df['home_team'].map(fpl_scores).fillna(50)
    df['fpl_away'] = df['away_team'].map(fpl_scores).fillna(50)
    df['fpl_diff'] = df['fpl_home'] - df['fpl_away']

    # Elo-derived probabilities
    df['elo_prob_home'] = 1 / (1 + 10**((df['Away_Elo'] - df['Home_Elo']) / 400))
    df['elo_prob_away'] = 1 / (1 + 10**((df['Home_Elo'] - df['Away_Elo']) / 400))
    df['elo_prob_draw'] = (1 - df['elo_prob_home'] - df['elo_prob_away']).clip(0.1, 0.4)

    # 4. For 2024 test set: use REAL betting odds instead of Elo proxy
    crowd_df['home_norm'] = crowd_df['HomeTeam'].apply(normalize_team_name)
    crowd_df['away_norm'] = crowd_df['AwayTeam'].apply(normalize_team_name)
    crowd_df['match_key'] = crowd_df['home_norm'] + ' vs ' + crowd_df['away_norm']
    crowd_df['total_prob'] = 1/crowd_df['HomeOdd'] + 1/crowd_df['DrawOdd'] + 1/crowd_df['AwayOdd']
    crowd_df['odds_h'] = (1/crowd_df['HomeOdd']) / crowd_df['total_prob']
    crowd_df['odds_d'] = (1/crowd_df['DrawOdd']) / crowd_df['total_prob']
    crowd_df['odds_a'] = (1/crowd_df['AwayOdd']) / crowd_df['total_prob']

    test_df = df[df['season'] == 2024].copy()
    test_df['match_key'] = test_df['home_team'] + ' vs ' + test_df['away_team']
    test_df = test_df.merge(
        crowd_df[['match_key', 'odds_h', 'odds_d', 'odds_a', 'CrowdPrediction']],
        on='match_key', how='left'
    )
    # Override elo proxy with real odds for testing
    test_df['elo_prob_home'] = test_df['odds_h'].fillna(test_df['elo_prob_home'])
    test_df['elo_prob_draw'] = test_df['odds_d'].fillna(test_df['elo_prob_draw'])
    test_df['elo_prob_away'] = test_df['odds_a'].fillna(test_df['elo_prob_away'])

    train_df = df[df['season'] != 2024]
    
    X_train = train_df[HYBRID_FEATURES]
    y_train = train_df['target']
    X_test = test_df[HYBRID_FEATURES].dropna()
    y_test = test_df.loc[X_test.index, 'target']
    
    return X_train, y_train, X_test, y_test, test_df.loc[X_test.index]

def train_stacking_v5():
    X_train, y_train, X_test, y_test, test_eval_df = prepare_data()
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set:  {X_test.shape}")
    
    # Define Base Learners
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ]
    
    # Define Meta-Learner (using a slightly more regularized LR)
    meta_learner = LogisticRegression(C=0.1)
    
    # Build Stacking Classifier
    # Use 5-fold CV to generate meta-features
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        n_jobs=-1
    )
    
    print("\nFitting Stacking Ensemble V5...")
    stack_model.fit(X_train, y_train)
    
    # Evaluate
    preds = stack_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- Stacking Ensemble V5 Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Comparison to Crowd (Odds)
    label_map = {0: 'HOME_TEAM', 1: 'DRAW', 2: 'AWAY_TEAM'}
    crowd_correct = (test_eval_df['CrowdPrediction'] == test_eval_df['target'].map(label_map)).sum()
    crowd_acc = crowd_correct / len(test_eval_df)
    
    print(f"Crowd (Odds) Accuracy: {crowd_acc*100:.2f}%")
    print(f"AI Improvement: {(acc - crowd_acc)*100:+.2f}%")
    
    # Save Model and Metrics
    os.makedirs('models', exist_ok=True)
    joblib.dump(stack_model, 'models/stacking_ensemble_v5_new.pkl')
    
    metrics = {
        "model": "Stacking Ensemble v5 (GB+XGB+RF\u2192LR)",
        "accuracy": round(acc * 100, 2),
        "features": HYBRID_FEATURES,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "crowd_accuracy": round(crowd_acc * 100, 2),
        "improvement_over_crowd": round((acc - crowd_acc) * 100, 2)
    }
    
    with open('models/metrics_v5_new.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nModel saved to models/stacking_ensemble_v5_new.pkl")
    return acc

if __name__ == "__main__":
    train_stacking_v5()
