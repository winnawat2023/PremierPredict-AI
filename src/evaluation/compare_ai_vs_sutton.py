import pandas as pd
import joblib
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def main():
    print("--- AI vs Chris Sutton Comparison ---")
    
    # 1. Load Data
    sutton_path = 'data/raw/sutton_predictions_2024.json'
    matches_path = 'data/processed/features.csv' 
    
    if not os.path.exists(sutton_path) or not os.path.exists(matches_path):
        print("Missing data files.")
        return

    with open(sutton_path, 'r') as f:
        sutton_data = json.load(f)
        
    matches_df = pd.read_csv(matches_path)
    # Filter for 2024
    matches_2024 = matches_df[matches_df['season'] == 2024].copy()
    
    # Load Model
    model = joblib.load('models/random_forest_v4.pkl')
    
    # Prepare Features
    feature_cols = [
        'Home_Form_L5', 'Away_Form_L5',
        'Home_Avg_GF_L5', 'Home_Avg_GA_L5',
        'Away_Avg_GF_L5', 'Away_Avg_GA_L5',
        'Home_MV', 'Away_MV', 'MV_Diff',
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'H2H_Home_Wins', 'H2H_Away_Wins'
    ]
    
    # Validate missing features? Assuming they exist
    X_test = matches_2024[feature_cols].fillna(0) # Basic fill
    ai_preds = model.predict(X_test)
    matches_2024['ai_pred'] = ai_preds 
    
    # Iterate through Sutton's list
    correct_sutton = 0
    correct_ai = 0
    total = 0
    
    print(f"\nEvaluating {len(sutton_data)} predictions from Sutton...")
    
    for item in sutton_data:
        home = item['home_team']
        away = item['away_team']
        sutton_pick = item['prediction_winner']
        
        # Find Actual Result
        match_row = matches_2024[
            (matches_2024['home_team'] == home) & 
            (matches_2024['away_team'] == away)
        ]
        
        if match_row.empty:
            print(f"Match not found in data: {home} vs {away}")
            continue
            
        actual = match_row.iloc[0]['winner']
        ai_p = match_row.iloc[0]['ai_pred']
        
        if ai_p == 0: ai_pick = 'HOME_TEAM'
        elif ai_p == 1: ai_pick = 'DRAW'
        else: ai_pick = 'AWAY_TEAM'
        
        total += 1
        
        if sutton_pick == actual:
            correct_sutton += 1
        
        if ai_pick == actual:
            correct_ai += 1
            
        print(f"{home} vs {away}: Real={actual}, Sutton={sutton_pick}, AI={ai_pick}")
        
    if total > 0:
        print(f"\n--- Results (N={total}) ---")
        print(f"Chris Sutton (Expert):   {correct_sutton/total*100:.2f}%")
        print(f"AI Model (v4):           {correct_ai/total*100:.2f}%")
    else:
        print("No matches evaluated.")

if __name__ == "__main__":
    main()
