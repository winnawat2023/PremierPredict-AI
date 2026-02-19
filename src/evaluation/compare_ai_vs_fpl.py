import pandas as pd
import joblib
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features import process_features

def main():
    print("--- AI vs FPL Baselines Comparison ---")
    
    # 1. Load Data
    crowd_path = 'data/raw/crowd_predictions_2024.csv'
    fpl_path = 'data/raw/fpl_baselines_2024.csv'
    matches_path = 'data/processed/features.csv' 
    
    if not os.path.exists(crowd_path) or not os.path.exists(fpl_path) or not os.path.exists(matches_path):
        print("Missing data files. Please run collection scripts first.")
        return

    crowd_df = pd.read_csv(crowd_path)
    fpl_df = pd.read_csv(fpl_path)
    matches_df = pd.read_csv(matches_path)
    
    # Filter matches for 2024 season only
    matches_2024 = matches_df[matches_df['season'] == 2024].copy()
    
    # Create Map for FPL Data
    # Name -> {strength_h, strength_a, ownership}
    fpl_map = fpl_df.set_index('name').to_dict('index')
    
    # Metrics
    correct_ai = 0
    correct_crowd = 0
    correct_fpl_official = 0
    correct_fpl_fan = 0
    total = 0
    
    results = []
    
    # Pre-load AI (already in matches_df predictions? No, we need to predict)
    # Load Model
    model = joblib.load('models/random_forest_v4.pkl')
    
    # Prepare Features for AI
    feature_cols = [
        'Home_Form_L5', 'Away_Form_L5',
        'Home_Avg_GF_L5', 'Home_Avg_GA_L5',
        'Away_Avg_GF_L5', 'Away_Avg_GA_L5',
        'Home_MV', 'Away_MV', 'MV_Diff',
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'H2H_Home_Wins', 'H2H_Away_Wins'
    ]
    
    X_test = matches_2024[feature_cols]
    ai_preds = model.predict(X_test)
    matches_2024['ai_pred'] = ai_preds # 0=Home, 1=Draw, 2=Away
    
    # Normalize Crowd Team Names (just in case)
    name_replacements = {
        'Man United': 'Manchester United FC',
        'Manchester Utd': 'Manchester United FC',
        'Man Utd': 'Manchester United FC',
        'Man City': 'Manchester City FC',
        'Tottenham': 'Tottenham Hotspur FC',
        'Spurs': 'Tottenham Hotspur FC',
        'Newcastle': 'Newcastle United FC',
        'West Ham': 'West Ham United FC',
        'Wolves': 'Wolverhampton Wanderers FC',
        'Brighton': 'Brighton & Hove Albion FC',
        'Leicester': 'Leicester City FC',
        'Leeds': 'Leeds United FC',
        'Nott\'m Forest': 'Nottingham Forest FC',
        'Sheffield Utd': 'Sheffield United FC',
        'Luton': 'Luton Town FC',
        'Arsenal': 'Arsenal FC',
        'Aston Villa': 'Aston Villa FC',
        'Bournemouth': 'AFC Bournemouth',
        'Brentford': 'Brentford FC',
        'Chelsea': 'Chelsea FC',
        'Crystal Palace': 'Crystal Palace FC',
        'Everton': 'Everton FC',
        'Fulham': 'Fulham FC',
        'Ipswich': 'Ipswich Town FC',
        'Liverpool': 'Liverpool FC',
        'Southampton': 'Southampton FC'
    }
    crowd_df['HomeTeam'] = crowd_df['HomeTeam'].replace(name_replacements)
    crowd_df['AwayTeam'] = crowd_df['AwayTeam'].replace(name_replacements)
    
    # DEBUG: Check names
    ai_teams = sorted(matches_2024['home_team'].unique())
    crowd_teams = sorted(crowd_df['HomeTeam'].unique())
    print("AI Teams:", ai_teams)
    print("Crowd Teams:", crowd_teams)
    
    # Iterate through matches
    print(f"Processing {len(matches_2024)} matches from AI Features...")
    
    for idx, row in matches_2024.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']
        actual = row['winner'] # HOME_TEAM, DRAW, AWAY_TEAM
        
        # 1. Crowd (Merge)
        # Find matching row in crowd_df
        crowd_row = crowd_df[(crowd_df['HomeTeam'] == home) & (crowd_df['AwayTeam'] == away)]
        
        if not crowd_row.empty:
            crowd_pred = crowd_row.iloc[0]['CrowdPrediction']
        else:
            # Debug: Try fuzzy match or just print
            # print(f"Missing Crowd Data for {home} vs {away}")
            crowd_pred = 'UNKNOWN'
            
        # 2. FPL Official
        h_stats = fpl_map.get(home, {'fpl_strength_home': 1000, 'fan_ownership_score': 0})
        a_stats = fpl_map.get(away, {'fpl_strength_away': 1000, 'fan_ownership_score': 0})
        
        # Logic
        h_str = h_stats['fpl_strength_home']
        a_str = a_stats['fpl_strength_away']
        
        if h_str > a_str + 50:
            fpl_off_pred = 'HOME_TEAM'
        elif a_str > h_str + 50:
            fpl_off_pred = 'AWAY_TEAM'
        else:
            fpl_off_pred = 'DRAW' 
            
        # 3. FPL Fan (Ownership)
        h_own = h_stats['fan_ownership_score']
        a_own = a_stats['fan_ownership_score']
        
        # Logic
        if h_own > a_own:
            fpl_fan_pred = 'HOME_TEAM'
        else:
            fpl_fan_pred = 'AWAY_TEAM'
            
        # 4. AI
        ai_p = row['ai_pred']
        if ai_p == 0: ai_pred_str = 'HOME_TEAM'
        elif ai_p == 1: ai_pred_str = 'DRAW'
        else: ai_pred_str = 'AWAY_TEAM'
        
        # Evaluate
        if crowd_pred != 'UNKNOWN':
            total += 1
            if crowd_pred == actual: correct_crowd += 1
            if fpl_off_pred == actual: correct_fpl_official += 1
            if fpl_fan_pred == actual: correct_fpl_fan += 1
            if ai_pred_str == actual: correct_ai += 1
        
    # Report
    if total > 0:
        print(f"\n--- Results (N={total} matches matched with Crowd Data) ---")
        print(f"1. Crowd (Betting Odds):     {correct_crowd/total*100:.2f}%")
        print(f"2. AI (Expert Model v4):     {correct_ai/total*100:.2f}%")
        print(f"3. FPL Official Strength:    {correct_fpl_official/total*100:.2f}%")
        print(f"4. FPL Fan Confidence:       {correct_fpl_fan/total*100:.2f}%")
        
        # Win Rates
        print("\n--- Insight ---")
        if correct_fpl_fan > correct_ai:
            print("Fans know best! Ownership is a better predictor.")
        else:
            print("AI beats Fan Popularity.")
    else:
        print("\nERROR: No matches aligned between AI data and Crowd/FPL data. Check Team Names.")
        print("Example AI Name:", matches_2024.iloc[0]['home_team'])
        print("Example Crowd Name:", crowd_df.iloc[0]['HomeTeam'])

if __name__ == "__main__":
    main()
