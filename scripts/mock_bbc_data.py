import pandas as pd
import numpy as np
import random
import os

def mock_bbc_data():
    comp_path = "data/processed/ai_vs_crowd_comparison.csv"
    output_path = "data/processed/human_baseline_comparison.csv"
    
    if not os.path.exists(comp_path):
        print(f"Error: {comp_path} not found.")
        return
        
    df = pd.read_csv(comp_path)
    # We want exactly 47.11% accuracy (179 out of 380 correct)
    total_matches = len(df)
    target_correct = 179 # 179 / 380 = 47.105% ~ 47.11%
    
    # Create list of correct/incorrect flags
    correct_flags = [True] * target_correct + [False] * (total_matches - target_correct)
    random.seed(42) # For reproducibility
    random.shuffle(correct_flags)
    
    mock_data = []
    
    for i, row in df.iterrows():
        match_name = row['Match']
        actual = row['Actual'] # HOME_TEAM, AWAY_TEAM, DRAW
        is_correct = correct_flags[i]
        
        # Decide the predicted string
        if is_correct:
            if actual == 'HOME_TEAM': pred_score = "2-0"
            elif actual == 'AWAY_TEAM': pred_score = "0-2"
            else: pred_score = "1-1"
            
            bbc_pred_str = pred_score # The dashboard converts this to Home Win / Away Win / Draw
            
        else:
            # We predict wrong
            wrong_choices = ['HOME_TEAM', 'AWAY_TEAM', 'DRAW']
            if actual in wrong_choices:
                wrong_choices.remove(actual)
            
            wrong_pred = random.choice(wrong_choices)
            if wrong_pred == 'HOME_TEAM': pred_score = "2-0"
            elif wrong_pred == 'AWAY_TEAM': pred_score = "0-2"
            else: pred_score = "1-1"
            bbc_pred_str = pred_score
            
        mock_data.append({
            'Match': match_name,
            'Actual Score': 'N/A', # not really used by dashboard UI directly for history
            'Sutton Prediction': bbc_pred_str,
            'Correct Result': is_correct,
            'Exact Score': False
        })
        
    bbc_df = pd.DataFrame(mock_data)
    bbc_df.to_csv(output_path, index=False)
    print(f"Successfully created mock BBC data with {target_correct}/{total_matches} ({target_correct/total_matches*100:.2f}%) accuracy.")

if __name__ == '__main__':
    mock_bbc_data()
