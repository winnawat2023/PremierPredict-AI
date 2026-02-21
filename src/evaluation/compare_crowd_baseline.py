import pandas as pd
import numpy as np
import os
from src.utils import normalize_team_name

def load_data():
    actual_path = 'data/raw/pl_matches_2021_2025.csv'
    crowd_path = 'data/raw/crowd_predictions_2024.csv'
    
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"{actual_path} not found.")
    if not os.path.exists(crowd_path):
        raise FileNotFoundError(f"{crowd_path} not found.")
        
    df_actual = pd.read_csv(actual_path)
    df_crowd = pd.read_csv(crowd_path)
    
    return df_actual, df_crowd

def compare_crowd(df_actual, df_crowd):
    print(f"Comparing {len(df_crowd)} crowd predictions against actual results...")
    
    # Normalize names in crowd data
    df_crowd['HomeTeam_Norm'] = df_crowd['HomeTeam'].apply(normalize_team_name)
    df_crowd['AwayTeam_Norm'] = df_crowd['AwayTeam'].apply(normalize_team_name)
    
    # Create match keys
    df_crowd['match_key'] = df_crowd['HomeTeam_Norm'] + " vs " + df_crowd['AwayTeam_Norm']
    
    # Filter actual data for 2024 season
    df_actual_2024 = df_actual[df_actual['season'] == 2024].copy()
    df_actual_2024['match_key'] = df_actual_2024['home_team'] + " vs " + df_actual_2024['away_team']
    
    # Merge
    merged = pd.merge(df_crowd, df_actual_2024, on='match_key', suffixes=('_crowd', '_actual'), how='inner')
    
    if merged.empty:
        print("No matching matches found.")
        print("Sample Crowd Keys:", df_crowd['match_key'].head().tolist())
        print("Sample Actual Keys:", df_actual_2024['match_key'].head().tolist())
        return

    correct = 0
    results = []
    
    for _, row in merged.iterrows():
        crowd_pred = row['CrowdPrediction'] # HOME_TEAM, AWAY_TEAM, DRAW
        actual_res = row['winner'] # HOME_TEAM, AWAY_TEAM, DRAW
        
        is_correct = (crowd_pred == actual_res)
        if is_correct:
            correct += 1
            
        results.append({
            'Match': row['match_key'],
            'Crowd Prediction': crowd_pred,
            'Actual Result': actual_res,
            'Correct': is_correct,
            'Home Odds': row['HomeOdd'],
            'Draw Odds': row['DrawOdd'],
            'Away Odds': row['AwayOdd']
        })
        
    accuracy = (correct / len(merged)) * 100
    print("\n--- Crowd/Market Baseline Performance ---")
    print(f"Matches Analyzed: {len(merged)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save processed comparison
    pd.DataFrame(results).to_csv('data/processed/crowd_baseline_comparison.csv', index=False)
    print("Saved comparison to data/processed/crowd_baseline_comparison.csv")

if __name__ == "__main__":
    try:
        df_actual, df_crowd = load_data()
        compare_crowd(df_actual, df_crowd)
    except Exception as e:
        print(f"Error: {e}")
