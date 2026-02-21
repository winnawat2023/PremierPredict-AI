"""
Ensemble Stacking Model v2 (Inspired by Beal et al., AAAI 2021)

HYBRID APPROACH: Augment the original 14 AI features with 6 human-signal features:
  - FPL Fan Ownership (home, away, diff)
  - Elo-derived match probabilities (home, draw, away)
  Total: 20 features → Gradient Boosting Classifier

Result: 53.95% accuracy (up from 51.32% base AI)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import normalize_team_name
from src.constants import BASE_FEATURES, HYBRID_FEATURES
# Features are now imported from src.constants


def build_ensemble():
    print("=" * 60)
    print("ENSEMBLE MODEL v2 (Hybrid: Stats + Human Signals)")
    print("=" * 60)

    # 1. Load data
    features_df = pd.read_csv('data/processed/features.csv')
    crowd_df = pd.read_csv('data/raw/crowd_predictions_2024.csv')
    fpl_df = pd.read_csv('data/raw/fpl_baselines_2024.csv')

    # 2. Prepare FPL scores
    fpl_df['name_norm'] = fpl_df['name'].apply(normalize_team_name)
    fpl_scores = dict(zip(fpl_df['name_norm'], fpl_df['fan_ownership_score']))

    # 3. Augment ALL data with human-signal features
    df = features_df.copy().dropna(subset=BASE_FEATURES)
    df['fpl_home'] = df['home_team'].map(fpl_scores).fillna(50)
    df['fpl_away'] = df['away_team'].map(fpl_scores).fillna(50)
    df['fpl_diff'] = df['fpl_home'] - df['fpl_away']

    # Elo-derived probabilities (proxy for odds in historical data)
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
    # Override elo proxy with real odds
    test_df['elo_prob_home'] = test_df['odds_h'].fillna(test_df['elo_prob_home'])
    test_df['elo_prob_draw'] = test_df['odds_d'].fillna(test_df['elo_prob_draw'])
    test_df['elo_prob_away'] = test_df['odds_a'].fillna(test_df['elo_prob_away'])

    train_df = df[df['season'] != 2024]

    # 5. Train Gradient Boosting model
    X_train = train_df[HYBRID_FEATURES]
    y_train = train_df['target']
    X_test = test_df[HYBRID_FEATURES].dropna()
    y_test = test_df.loc[X_test.index, 'target']

    print(f"\nTraining: {len(X_train)} matches ({sorted(train_df['season'].unique())})")
    print(f"Testing:  {len(X_test)} matches (Season 2024)")
    print(f"Features: {len(HYBRID_FEATURES)} ({len(BASE_FEATURES)} base + 6 human)")

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # 6. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Crowd accuracy
    label_map = {0: 'HOME_TEAM', 1: 'DRAW', 2: 'AWAY_TEAM'}
    test_eval = test_df.loc[X_test.index].copy()
    crowd_correct = (test_eval['CrowdPrediction'] == test_eval['target'].map(label_map)).sum()
    crowd_acc = crowd_correct / len(test_eval)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Ensemble v2 (GB Hybrid): {acc*100:.2f}%")
    print(f"  Crowd (Betting Odds):    {crowd_acc*100:.2f}%")
    print(f"{'='*60}")

    # Feature importance
    imp = sorted(zip(HYBRID_FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print(f"\n--- Feature Importance ---")
    for f, v in imp:
        bar = '█' * int(v * 100)
        print(f"  {f:20s} {v:.3f} {bar}")

    # 7. Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/ensemble_v2_gb.pkl')
    print(f"\nModel saved: models/ensemble_v2_gb.pkl")

    # Save results
    results = {
        'ensemble_accuracy': round(acc * 100, 2),
        'crowd_accuracy': round(crowd_acc * 100, 2),
        'model_type': 'GradientBoosting (Hybrid)',
        'n_features': len(HYBRID_FEATURES),
        'features': HYBRID_FEATURES,
    }
    with open('data/processed/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return acc


if __name__ == "__main__":
    build_ensemble()
