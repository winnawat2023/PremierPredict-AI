import pandas as pd
import numpy as np
import os

def calculate_points(result):
    if result == 'win':
        return 3
    elif result == 'draw':
        return 1
    else:
        return 0

def get_last_5_form(past_matches):
    """
    Calculate points from last 5 matches.
    past_matches: list of 'win', 'draw', 'loss' results
    """
    if not past_matches:
        return 0
    # Take last 5
    last_5 = past_matches[-5:]
    points = sum([calculate_points(r) for r in last_5])
    return points / 5.0 # Average points

def process_features(input_path="data/raw/pl_matches_2021_2025.csv", output_path="data/processed/features.csv"):
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    # Sort by date to ensure correct historical calculation
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Initialize team stats tracking
    # Dictionary to store team state: { 'TeamName': { 'points': 0, 'games': 0, 'history': [], 'goal_diff': 0, 'goals_for': 0 } }
    team_stats = {}
    
    # Features lists to append to DataFrame
    home_form_l5 = []
    away_form_l5 = []
    home_position = []
    away_position = []
    position_diff = []
    
    # Pre-fill team stats with 0 for all teams found in the dataset
    all_teams = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    for team in all_teams:
        team_stats[team] = {'points': 0, 'games': 0, 'history': [], 'goal_diff': 0, 'goals_for': 0}
        
    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']
        
        # Reset stats if it's a new season
        # Check if we moved to a new season year compared to previous row (or just handle per row)
        # We need to detect season change.
        if index > 0:
            prev_season = df.iloc[index-1]['season']
            if season != prev_season:
                 # Reset all stats for next season
                 for team in all_teams:
                    team_stats[team] = {'points': 0, 'games': 0, 'history': [], 'goal_diff': 0, 'goals_for': 0}

        
        # Calculate Features BEFORE the match (based on historical data)
        h_stats = team_stats[home]
        a_stats = team_stats[away]
        
        # Form L5
        h_form = get_last_5_form(h_stats['history'])
        a_form = get_last_5_form(a_stats['history'])
        home_form_l5.append(h_form)
        away_form_l5.append(a_form)
        
        # Position Calculation
        # Create a temporary dataframe of current standings
        standings = []
        for t, stats in team_stats.items():
            # Only include teams that have played or are in the current season context?
            # Ideally we keep all teams, points start at 0.
             standings.append({
                 'team': t,
                 'points': stats['points'],
                 'gd': stats['goal_diff'],
                 'gf': stats['goals_for']
             })
        standings_df = pd.DataFrame(standings)
        # Sort by Points (Desc), Goal Diff (Desc), Goals For (Desc)
        standings_df = standings_df.sort_values(by=['points', 'gd', 'gf'], ascending=[False, False, False]).reset_index(drop=True)
        standings_df['rank'] = standings_df.index + 1
        
        h_rank = standings_df.loc[standings_df['team'] == home, 'rank'].values[0]
        a_rank = standings_df.loc[standings_df['team'] == away, 'rank'].values[0]
        
        home_position.append(h_rank)
        away_position.append(a_rank)
        # Position Diff: Home Rank - Away Rank
        # If Home is #1 and Away is #5, Diff is -4.
        # If Home is #10 and Away is #2, Diff is 8.
        position_diff.append(h_rank - a_rank) 
        
        # Update Stats AFTER the match
        winner = row['winner']
        home_score = row['home_score']
        away_score = row['away_score']
        
        # Update games
        team_stats[home]['games'] += 1
        team_stats[away]['games'] += 1
        
        # Update goals
        team_stats[home]['goals_for'] += home_score
        team_stats[away]['goals_for'] += away_score
        team_stats[home]['goal_diff'] += (home_score - away_score)
        team_stats[away]['goal_diff'] += (away_score - home_score)
        
        if winner == 'HOME_TEAM':
            team_stats[home]['points'] += 3
            team_stats[home]['history'].append('win')
            team_stats[away]['history'].append('loss')
        elif winner == 'AWAY_TEAM':
            team_stats[away]['points'] += 3
            team_stats[away]['history'].append('win')
            team_stats[home]['history'].append('loss')
        else: # DRAW
            team_stats[home]['points'] += 1
            team_stats[away]['points'] += 1
            team_stats[home]['history'].append('draw')
            team_stats[away]['history'].append('draw')
            

    # Add features to DF
    df['Home_Form_L5'] = home_form_l5
    df['Away_Form_L5'] = away_form_l5
    df['Home_Pos'] = home_position
    df['Away_Pos'] = away_position
    df['Position_Diff'] = position_diff # Home - Away
    # Home Advantage is implicit in model or we can add explicit flag if needed, usually just Home/Away designation is enough, 
    # but Master Brief says "Home_Advantage (Binary or calculated)".
    # Let's add Home_Advantage = 1 for all rows since it's from Home team perspective?
    # Or maybe it's a fixed value. For now let's just use the fact that it is a classification of Home Win.
    # We can add a dummy 'Home_Advantage' column = 1.
    df['Home_Advantage'] = 1
    
    # Target Mapping: 0: Home Win, 1: Draw, 2: Away Win
    label_map = {'HOME_TEAM': 0, 'DRAW': 1, 'AWAY_TEAM': 2}
    df['target'] = df['winner'].map(label_map)
    
    # Drop rows where target is NaN (if any)
    df.dropna(subset=['target'], inplace=True)
    
    # Select final columns
    final_cols = ['date', 'home_team', 'away_team', 'season', 'Home_Advantage', 'Home_Form_L5', 'Away_Form_L5', 'Position_Diff', 'target']
    result_df = df[final_cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Features processed and saved to {output_path}. Shape: {result_df.shape}")

if __name__ == "__main__":
    process_features()
