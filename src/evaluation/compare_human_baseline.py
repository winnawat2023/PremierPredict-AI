import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load actual results and Sutton predictions."""
    actual_path = 'data/raw/pl_matches_2021_2025.csv'
    sutton_path = 'data/raw/sutton_predictions_2024_2025.csv'
    
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"{actual_path} not found.")
    if not os.path.exists(sutton_path):
        raise FileNotFoundError(f"{sutton_path} not found.")
        
    df_actual = pd.read_csv(actual_path)
    df_sutton = pd.read_csv(sutton_path)
    
    return df_actual, df_sutton

def get_match_result(home_score, away_score):
    if home_score > away_score:
        return 'HOME_TEAM'
    elif away_score > home_score:
        return 'AWAY_TEAM'
    else:
        return 'DRAW'

def compare_predictions(df_actual, df_sutton):
    print(f"Comparing {len(df_sutton)} predictions against actual results...")
    
    # Merge on home_team and away_team
    # Note: dates might not match perfectly if scraped from articles vs official data, 
    # so we rely on team names which must be unique per season usually 
    # (but they play twice, so we need to be careful).
    # Ideally we'd filter by season too, but for now we assume sutton data is current season (2024-2025).
    
    # Filter actual data for 2024 season (which spans 2024-2025)
    # The 'season' column in pl_matches usually refers to start year, e.g., 2024.
    df_actual_2024 = df_actual[df_actual['season'] == 2024].copy()
    
    # Create a key for merging to handle home/away fixtures
    df_actual_2024['match_key'] = df_actual_2024['home_team'] + " vs " + df_actual_2024['away_team']
    df_sutton['match_key'] = df_sutton['home_team'] + " vs " + df_sutton['away_team']
    
    merged = pd.merge(df_sutton, df_actual_2024, on='match_key', suffixes=('_sutton', '_actual'), how='inner')
    
    if merged.empty:
        print("No matching matches found between predictions and actual results.")
        # Debugging
        print("Sample Actual Keys:", df_actual_2024['match_key'].head().tolist())
        print("Sample Sutton Keys:", df_sutton['match_key'].head().tolist())
        return

    # Calculate Accuracy
    correct_results = 0
    exact_scores = 0
    
    results = []
    
    for _, row in merged.iterrows():
        # Actual Result
        # Using the 'winner' column from actual data which is HOME_TEAM, AWAY_TEAM, DRAW
        actual_result = row['winner']
        
        # Sutton Result
        sutton_result = get_match_result(row['sutton_home_score'], row['sutton_away_score'])
        
        # Check Result Accuracy
        is_correct_result = (sutton_result == actual_result)
        if is_correct_result:
            correct_results += 1
            
        # Check Exact Score
        is_exact_score = (row['sutton_home_score'] == row['home_score']) and \
                         (row['sutton_away_score'] == row['away_score'])
        if is_exact_score:
            exact_scores += 1
            
        results.append({
            'Match': row['match_key'],
            'Actual Score': f"{row['home_score']}-{row['away_score']}",
            'Sutton Prediction': f"{row['sutton_home_score']}-{row['sutton_away_score']}",
            'Correct Result': is_correct_result,
            'Exact Score': is_exact_score
        })
    
    df_results = pd.DataFrame(results)
    
    accuracy_result = (correct_results / len(merged)) * 100
    accuracy_exact = (exact_scores / len(merged)) * 100
    
    print("\n--- Human Baseline (Chris Sutton) Performance ---")
    print(f"Matches Analyzed: {len(merged)}")
    print(f"Correct Match Result Accuracy: {accuracy_result:.2f}%")
    print(f"Exact Score Accuracy: {accuracy_exact:.2f}%")
    
    print("\nDetailed Comparison:")
    print(df_results)
    
    # Save results
    df_results.to_csv('data/processed/human_baseline_comparison.csv', index=False)
    print("\nComparison details saved to data/processed/human_baseline_comparison.csv")

if __name__ == "__main__":
    try:
        df_actual, df_sutton = load_data()
        compare_predictions(df_actual, df_sutton)
    except Exception as e:
        print(f"Error: {e}")
