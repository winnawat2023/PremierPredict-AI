import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data():
    features_path = 'data/processed/features.csv'
    crowd_path = 'data/raw/crowd_predictions_2024.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found.")
    if not os.path.exists(crowd_path):
        raise FileNotFoundError(f"{crowd_path} not found.")
        
    df_features = pd.read_csv(features_path)
    df_crowd = pd.read_csv(crowd_path)
    
    return df_features, df_crowd

def normalize_crowd_names(name):
    # Mapping crowd data names (Bet365) to our feature names (likely full names)
    # Our features use names like 'Manchester United FC', crowd uses 'Man United'
    mapping = {
        'Man United': 'Manchester United FC',
        'Man Utd': 'Manchester United FC',
        'Man City': 'Manchester City FC',
        'Liverpool': 'Liverpool FC',
        'Arsenal': 'Arsenal FC',
        'Aston Villa': 'Aston Villa FC',
        'Tottenham': 'Tottenham Hotspur FC',
        'Chelsea': 'Chelsea FC',
        'Newcastle': 'Newcastle United FC',
        'West Ham': 'West Ham United FC',
        'Brighton': 'Brighton & Hove Albion FC',
        'Brentford': 'Brentford FC',
        'Crystal Palace': 'Crystal Palace FC',
        'Wolves': 'Wolverhampton Wanderers FC',
        'Fulham': 'Fulham FC',
        'Bournemouth': 'AFC Bournemouth',
        'Everton': 'Everton FC',
        'Nott\'m Forest': 'Nottingham Forest FC',
        'Luton': 'Luton Town FC',
        'Burnley': 'Burnley FC',
        'Sheffield United': 'Sheffield United FC',
        'Leicester': 'Leicester City FC',
        'Ipswich': 'Ipswich Town FC',
        'Southampton': 'Southampton FC',
        'Leeds': 'Leeds United FC',
        # Handle cases where names might be strictly equal already or slight variations
        'Ipswich Town': 'Ipswich Town FC',
        'Leicester City': 'Leicester City FC',
        'Nottingham Forest': 'Nottingham Forest FC'
    }
    return mapping.get(name, name)

def compare_ai_vs_crowd(df_features, df_crowd):
    print("--- AI Model vs Crowd Baseline Comparison ---")
    
    # 1. Prepare Data for AI Model
    # Split strictly by season to simulate real-world usage (train on past, predict future)
    train_df = df_features[df_features['season'] != 2024]
    test_df = df_features[df_features['season'] == 2024].copy()
    
    train_seasons = sorted(train_df['season'].unique())
    print(f"Training Data (Seasons {train_seasons}): {len(train_df)} matches")
    print(f"Testing Data (2024 Season): {len(test_df)} matches")
    # Define Features
    features = [
        'Home_Form_L5', 'Away_Form_L5',
        'Home_Avg_GF_L5', 'Home_Avg_GA_L5',
        'Away_Avg_GF_L5', 'Away_Avg_GA_L5',
        'Home_MV', 'Away_MV', 'MV_Diff',
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'H2H_Home_Wins', 'H2H_Away_Wins'
    ]
    
    # Prepare X and y
    # Drop rows with NaN (e.g. first 5 matches where form is not fully calc)
    train_df = train_df.dropna(subset=features)
    test_df = test_df.dropna(subset=features)
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target'] # Ground truth from features file
    
    # 2. Train AI Model (with Tuning)
    print("Tuning Random Forest Classifier (GridSearchCV)...")
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"Best Parameters Found: {best_params}")
    
    model = grid_search.best_estimator_
    
    # Save the model
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_v4.pkl')
    print("Model saved to models/random_forest_v4.pkl")
    
    # 3. Generate Predictions
    ai_preds = model.predict(X_test)

    # Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("\n--- Feature Importance ---")
    print(feature_importance_df)
    test_df['ai_prediction'] = ai_preds
    
    # Map numeric target back to string for comparison with crowd
    # 0: Home Win, 1: Draw, 2: Away Win
    label_map_rev = {0: 'HOME_TEAM', 1: 'DRAW', 2: 'AWAY_TEAM'}
    test_df['ai_pred_label'] = test_df['ai_prediction'].map(label_map_rev)
    test_df['actual_label'] = test_df['target'].map(label_map_rev)
    
    # Calcluate AI Accuracy on full test set provided by features
    ai_acc_full = accuracy_score(y_test, ai_preds)
    print(f"AI Model Accuracy on all 2024 data: {ai_acc_full*100:.2f}%")
    
    # 4. Merge with Crowd Data to compare on EXACT subset
    # Normalize crowd names first
    df_crowd['HomeTeam_Norm'] = df_crowd['HomeTeam'].apply(normalize_crowd_names)
    df_crowd['AwayTeam_Norm'] = df_crowd['AwayTeam'].apply(normalize_crowd_names)
    
    # Create join keys
    df_crowd['match_key'] = df_crowd['HomeTeam_Norm'] + " vs " + df_crowd['AwayTeam_Norm']
    test_df['match_key'] = test_df['home_team'] + " vs " + test_df['away_team']
    
    # Merge
    comparison_df = pd.merge(test_df, df_crowd[['match_key', 'CrowdPrediction']], on='match_key', how='inner')
    
    print(f"\nMatches aligned for comparison: {len(comparison_df)}")
    
    if comparison_df.empty:
        print("Error: No matches matched between AI test set and Crowd data.")
        return

    # 5. Evaluate Matches
    ai_correct = 0
    crowd_correct = 0
    matches_total = len(comparison_df)
    
    comparison_details = []
    
    for _, row in comparison_df.iterrows():
        actual = row['actual_label']
        ai_pred = row['ai_pred_label']
        crowd_pred = row['CrowdPrediction']
        
        is_ai_correct = (ai_pred == actual)
        is_crowd_correct = (crowd_pred == actual)
        
        if is_ai_correct:
            ai_correct += 1
        if is_crowd_correct:
            crowd_correct += 1
            
        comparison_details.append({
            'Match': row['match_key'],
            'Actual': actual,
            'AI_Pred': ai_pred,
            'Crowd_Pred': crowd_pred,
            'AI_Correct': is_ai_correct,
            'Crowd_Correct': is_crowd_correct
        })
        
    ai_accuracy = (ai_correct / matches_total) * 100
    crowd_accuracy = (crowd_correct / matches_total) * 100
    
    print(f"\n--- Final Results ({matches_total} matches) ---")
    print(f"Crowd/Baseline Accuracy: {crowd_accuracy:.2f}%")
    print(f"AI Model Accuracy:       {ai_accuracy:.2f}%")
    
    diff = ai_accuracy - crowd_accuracy
    if diff > 0:
        print(f"\nRESULT: AI beats Crowd by +{diff:.2f}%!")
    else:
        print(f"\nRESULT: AI trails Crowd by {diff:.2f}%.")
        
    # Save details
    pd.DataFrame(comparison_details).to_csv('data/processed/ai_vs_crowd_comparison.csv', index=False)
    print("\nDetailed comparison saved to data/processed/ai_vs_crowd_comparison.csv")

if __name__ == "__main__":
    try:
        df_feat, df_crowd = load_data()
        compare_ai_vs_crowd(df_feat, df_crowd)
    except Exception as e:
        print(f"Error: {e}")
