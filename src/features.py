import pandas as pd
import numpy as np
import os
from difflib import get_close_matches

def calculate_points(result):
    if result == 'win':
        return 3
    elif result == 'draw':
        return 1
    else:
        return 0

def calculate_form(past_matches_history):
    """
    Calculate points from last 5 matches based on history dictionaries.
    past_matches_history: list of dicts like {'result': 'W', 'points': 3, ...}
    """
    if not past_matches_history:
        return 0.0
    # The input `past_matches_history` is already the last 5 (or fewer)
    points = sum([m['points'] for m in past_matches_history])
    return points / 5.0 # Average points

def load_market_values(filepath="data/raw/market_values.csv"):
    if not os.path.exists(filepath):
        print("Market value file not found. Using default 0.")
        return pd.DataFrame(columns=['season', 'club_name', 'market_value_m'])
    return pd.read_csv(filepath)

def get_best_match(name, options):
    matches = get_close_matches(name, options, n=1, cutoff=0.6)
    return matches[0] if matches else name

def calculate_rolling_stats(rows):
    # rows: list of (goals_for, goals_against)
    if not rows:
        return 0.0, 0.0
    
    total_gf = sum(r[0] for r in rows)
    total_ga = sum(r[1] for r in rows)
    return total_gf / len(rows), total_ga / len(rows)

def calculate_elo_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, actual_score_a, k=20):
    expected_a = calculate_elo_expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (actual_score_a - expected_a)
    return new_rating_a

def process_features(input_path="data/raw/all_matches_2020_2024.csv", output_path="data/processed/features.csv"):
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # NORMALIZE TEAM NAMES
    # Fix inconsistencies (Man United vs Manchester United FC vs Manchester Utd)
    name_replacements = {
        'Man United': 'Manchester United FC',
        'Manchester Utd': 'Manchester United FC',
        'Man Utd': 'Manchester United FC',
        'Manchester United': 'Manchester United FC',
        'Man City': 'Manchester City FC',
        'Manchester City': 'Manchester City FC',
        'Tottenham': 'Tottenham Hotspur FC',
        'Spurs': 'Tottenham Hotspur FC',
        'Newcastle': 'Newcastle United FC',
        'West Ham': 'West Ham United FC',
        'Wolves': 'Wolverhampton Wanderers FC',
        'Brighton': 'Brighton & Hove Albion FC',
        'Leicester': 'Leicester City FC',
        'Leeds': 'Leeds United FC',
        'Gardners': 'Luton Town FC', # Just in case
        'Sheffield Utd': 'Sheffield United FC',
        'Nott\'m Forest': 'Nottingham Forest FC'
    }
    df['home_team'] = df['home_team'].replace(name_replacements)
    df['away_team'] = df['away_team'].replace(name_replacements)

    # Load Market Values
    mv_df = load_market_values()
    
    # Normalize Team Names in MV to match Match Data
    # Get unique team names from match data
    match_teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    
    # Create a mapping for MV names -> Match names
    mv_teams = mv_df['club_name'].unique()
    mv_name_map = {}
    print("--- Team Name Mapping Debug ---")
    for mv_name in mv_teams:
        matches = get_close_matches(mv_name, match_teams, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            mv_name_map[mv_name] = best_match
            if mv_name != best_match:
                 print(f"Mapped MV '{mv_name}' -> Match '{best_match}'")
        else:
            print(f"WARNING: No match found for MV Team '{mv_name}'")
            mv_name_map[mv_name] = mv_name # Fallback
            
    mv_df['club_name'] = mv_df['club_name'].map(mv_name_map)
    
    features = []
    
    # Setup team history tracker
    team_history = {} # { team_name: [ {'result': 'W', 'points': 3, 'gf': 2, 'ga': 1}, ... ] }
    team_elo = {} # { team_name: 1500.0 }
    h2h_history = {} # { tuple(sorted((team_a, team_b))): [winner_name, ...] }

    # Filter out unplayed matches (NaN scores corrupt history)
    df = df.dropna(subset=['home_score', 'away_score', 'winner'])
    print(f"Processing {len(df)} matches (dropped unplayed/future matches)")

    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']
        
        # 1. Standard Features (Form & Points)
        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])
        
        home_form_l5 = calculate_form(home_hist[-5:])
        away_form_l5 = calculate_form(away_hist[-5:])
        
        # 2. Goal Stats (Rolling Avg L5)
        # Extract (gf, ga) tuples for calculation
        home_goals_l5 = [(m['gf'], m['ga']) for m in home_hist[-5:]]
        away_goals_l5 = [(m['gf'], m['ga']) for m in away_hist[-5:]]
        
        home_avg_gf, home_avg_ga = calculate_rolling_stats(home_goals_l5)
        away_avg_gf, away_avg_ga = calculate_rolling_stats(away_goals_l5)
        
        # 3. Market Value Lookup
        # Find MV for this season
        home_mv_row = mv_df[(mv_df['season'] == season) & (mv_df['club_name'] == home)]
        away_mv_row = mv_df[(mv_df['season'] == season) & (mv_df['club_name'] == away)]
        
        home_mv = home_mv_row['market_value_m'].values[0] if not home_mv_row.empty else 0
        away_mv = away_mv_row['market_value_m'].values[0] if not away_mv_row.empty else 0
        
        # 4. Elo Ratings
        home_elo = team_elo.get(home, 1500.0)
        away_elo = team_elo.get(away, 1500.0)
        
        # 5. H2H Features
        pair_key = tuple(sorted((home, away)))
        past_meetings = h2h_history.get(pair_key, [])
        last_5_meetings = past_meetings[-5:]
        
        h2h_home_wins = last_5_meetings.count(home)
        h2h_away_wins = last_5_meetings.count(away)
        h2h_draws = last_5_meetings.count('DRAW')
        
        features.append({
            'match_key': f"{row['date']}_{home}_{away}",
            'season': season,
            'home_team': home,
            'away_team': away,
            'Home_Form_L5': home_form_l5,
            'Away_Form_L5': away_form_l5,
            'Home_Avg_GF_L5': home_avg_gf,
            'Home_Avg_GA_L5': home_avg_ga,
            'Away_Avg_GF_L5': away_avg_gf,
            'Away_Avg_GA_L5': away_avg_ga,
            'Home_MV': home_mv,
            'Away_MV': away_mv,
            'MV_Diff': home_mv - away_mv,
            'Home_Elo': home_elo,
            'Away_Elo': away_elo,
            'Elo_Diff': home_elo - away_elo,
            'H2H_Home_Wins': h2h_home_wins,
            'H2H_Away_Wins': h2h_away_wins,
            'winner': row['winner']
        })
        
        # Update History & Elo & H2H
        winner = row['winner']
        home_res = 'D'
        away_res = 'D'
        home_pts = 1
        away_pts = 1
        
        # S_A for Elo (1=Win, 0.5=Draw, 0=Loss)
        home_s = 0.5
        away_s = 0.5
        match_winner_name = 'DRAW'
        
        if winner == 'HOME_TEAM':
            home_res = 'W'
            away_res = 'L'
            home_pts = 3
            away_pts = 0
            home_s = 1.0
            away_s = 0.0
            match_winner_name = home
        elif winner == 'AWAY_TEAM':
            home_res = 'L'
            away_res = 'W'
            home_pts = 0
            away_pts = 3
            home_s = 0.0
            away_s = 1.0
            match_winner_name = away
            
        if home not in team_history: team_history[home] = []
        if away not in team_history: team_history[away] = []
        
        team_history[home].append({'result': home_res, 'points': home_pts, 'gf': row['home_score'], 'ga': row['away_score']})
        team_history[away].append({'result': away_res, 'points': away_pts, 'gf': row['away_score'], 'ga': row['home_score']})
        
        # Update Elo
        new_home_elo = update_elo(home_elo, away_elo, home_s)
        new_away_elo = update_elo(away_elo, home_elo, away_s)
        team_elo[home] = new_home_elo
        team_elo[away] = new_away_elo
        
        # Update H2H
        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append(match_winner_name)
    features_df = pd.DataFrame(features)
    
    # Encode Target
    label_map = {'HOME_TEAM': 0, 'DRAW': 1, 'AWAY_TEAM': 2}
    features_df['target'] = features_df['winner'].map(label_map)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Features processed and saved to {output_path}. Shape: {features_df.shape}")
    
    # Save Latest Stats for Dashboard
    import json
    
    # helper to serialise
    def serialize_history(hist):
        # Keep last 5 only
        return hist[-5:]
        
    # Get latest MV for each team (use latest season IN mv_df, not match data)
    if not mv_df.empty:
        latest_mv_season = mv_df['season'].max()
        latest_mv_df = mv_df[mv_df['season'] == latest_mv_season]
        team_mv_map = dict(zip(latest_mv_df['club_name'], latest_mv_df['market_value_m']))
        print(f"Using Market Value from season {latest_mv_season} ({len(team_mv_map)} teams)")
    else:
        team_mv_map = {}
    
    # Convert H2H keys from tuple to string for JSON
    h2h_export = {f"{k[0]}_vs_{k[1]}": v for k, v in h2h_history.items()}
    
    final_stats = {
        'elo': team_elo,
        'history': {k: serialize_history(v) for k, v in team_history.items()},
        'h2h': h2h_export,
        'market_value': team_mv_map,
        'last_updated': str(pd.Timestamp.now())
    }
    
    stats_path = os.path.join(os.path.dirname(output_path), 'latest_team_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=4)
    print(f"Latest team stats saved to {stats_path}")

if __name__ == "__main__":
    process_features()
