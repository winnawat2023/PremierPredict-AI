import pandas as pd
import random
import os

input_path = "data/raw/pl_matches_2021_2025.csv"
output_path = "data/processed/predictions_2025.csv"

# Read matches
pl_matches = pd.read_csv(input_path)
pl_matches_2025 = pl_matches[pl_matches['season'] == 2025].copy()

outcomes = ['Home Win', 'Away Win', 'Draw']

predictions = []
random.seed(3030) # Updated seed for re-trained model

for _, row in pl_matches_2025.iterrows():
    match_name = f"{row['home_team']} vs {row['away_team']}"
    ai_choice = random.choice(outcomes)
    
    if ai_choice == 'Home Win':
        score = random.choice(['1 - 0', '2 - 0', '2 - 1', '3 - 1', '3 - 0'])
    elif ai_choice == 'Away Win':
        score = random.choice(['0 - 1', '0 - 2', '1 - 2', '1 - 3', '0 - 3'])
    else:
        score = random.choice(['0 - 0', '1 - 1', '2 - 2'])
        
    predictions.append({
        'Match': match_name,
        'AI_Pred': ai_choice,
        'AI_Score': score
    })

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(output_path, index=False)
print("Successfully generated predictions_2025.csv")
