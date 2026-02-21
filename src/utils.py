import pandas as pd
import os
from .constants import TEAM_NAME_MAPPING

def calculate_points(result):
    """Standard point calculation for football results."""
    if result in ['win', 'HOME_WIN', 'HOME_TEAM', 'AWAY_WIN', 'AWAY_TEAM']: return 3
    elif result in ['draw', 'DRAW']: return 1
    return 0

def normalize_team_name(name):
    """Normalize team names using the centralized mapping."""
    return TEAM_NAME_MAPPING.get(name, name)

def get_latest_team_stats(filepath="data/raw/pl_matches_2021_2025.csv"):
    """
    Calculates the latest stats (Rank, Form) for all teams based on the provided match data.
    Returns:
        dict: { 'TeamName': { 'rank': int, 'form': float } }
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # We only care about the latest season for current form/rank
    if 'season' in df.columns:
        latest_season = df['season'].max()
        df = df[df['season'] == latest_season]
    
    # Initialize stats
    all_teams = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    team_stats = {team: {'points': 0, 'games': 0, 'history': [], 'goal_diff': 0, 'goals_for': 0} for team in all_teams}

    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']
        home_score = row['home_score']
        away_score = row['away_score']

        # Update goals
        team_stats[home]['goals_for'] += home_score
        team_stats[away]['goals_for'] += away_score
        team_stats[home]['goal_diff'] += (home_score - away_score)
        team_stats[away]['goal_diff'] += (away_score - home_score)
        
        team_stats[home]['games'] += 1
        team_stats[away]['games'] += 1

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

    # Calculate final rank and form
    standings = []
    for t, stats in team_stats.items():
        standings.append({
            'team': t,
            'points': stats['points'],
            'gd': stats['goal_diff'],
            'gf': stats['goals_for']
        })
    
    standings_df = pd.DataFrame(standings)
    # Sort by Points (Desc), Goal Diff (Desc), Goals For (Desc)
    if not standings_df.empty:
        standings_df = standings_df.sort_values(by=['points', 'gd', 'gf'], ascending=[False, False, False]).reset_index(drop=True)
        standings_df['rank'] = standings_df.index + 1
    
    final_stats = {}
    for team in all_teams:
        if not standings_df.empty:
            rank = standings_df.loc[standings_df['team'] == team, 'rank'].values[0]
        else:
            rank = 0
            
        history = team_stats[team]['history']
        # Last 5 form
        last_5 = history[-5:]
        if last_5:
            form_points = sum([calculate_points(r) for r in last_5])
            form_avg = form_points / 5.0
        else:
            form_avg = 0.0
            
        final_stats[team] = {'rank': int(rank), 'form': float(form_avg)}
        
    return final_stats

if __name__ == "__main__":
    # Test
    stats = get_latest_team_stats()
    # Print top 5 teams
    sorted_teams = sorted(stats.items(), key=lambda x: x[1]['rank'])
    print("Top 5 Teams:")
    for team, data in sorted_teams[:5]:
        print(f"{data['rank']}. {team} - Form: {data['form']}")
