import pandas as pd
import requests
from io import StringIO
import os
from difflib import get_close_matches

def fetch_fpl_data():
    teams_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/teams.csv"
    players_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv"
    
    print("Fetching FPL Teams...")
    teams_df = pd.read_csv(teams_url)
    
    print("Fetching FPL Players...")
    players_df = pd.read_csv(players_url)
    
    return teams_df, players_df

def process_fpl_data(teams_df, players_df):
    # 1. Start with Team Strength
    teams_df = teams_df[['id', 'name', 'strength_overall_home', 'strength_overall_away']].copy()
    teams_df.rename(columns={'id': 'team_id', 'strength_overall_home': 'fpl_strength_home', 'strength_overall_away': 'fpl_strength_away'}, inplace=True)
    
    # 2. Aggregating Fan Confidence (Ownership)
    # Map player's team_code to team_id (Wait, players_df has 'team' column which matches team id)
    # Check 'team' column in players_raw.csv
    
    team_ownership = {}
    
    for team_id in teams_df['team_id']:
        # Get all players for this team
        team_players = players_df[players_df['team'] == team_id]
        
        # Sort by selected_by_percent (descending)
        team_players = team_players.sort_values(by='selected_by_percent', ascending=False)
        
        # Sum ownership of top 15 players (Squad size)
        # This gives a "Total Fan Interest" metric
        total_ownership = team_players['selected_by_percent'].head(15).sum()
        team_ownership[team_id] = total_ownership
        
    teams_df['fan_ownership_score'] = teams_df['team_id'].map(team_ownership)
    
    return teams_df

def normalize_names(df):
    # Standardize names to match PremierPredictAI
    # Current names in FPL: Arsenal, Aston Villa, etc.
    # Target names: Arsenal FC, Aston Villa FC, etc.
    
    # Load our standard names from existing data or define map
    # A simple map for now based on what we know
    name_map = {
        'Arsenal': 'Arsenal FC',
        'Aston Villa': 'Aston Villa FC', 
        'Bournemouth': 'AFC Bournemouth',
        'Brentford': 'Brentford FC',
        'Brighton': 'Brighton & Hove Albion FC',
        'Chelsea': 'Chelsea FC',
        'Crystal Palace': 'Crystal Palace FC',
        'Everton': 'Everton FC',
        'Fulham': 'Fulham FC',
        'Ipswich': 'Ipswich Town FC',
        'Leicester': 'Leicester City FC',
        'Liverpool': 'Liverpool FC',
        'Man City': 'Manchester City FC',
        'Man Utd': 'Manchester United FC',
        'Newcastle': 'Newcastle United FC',
        'Nott\'m Forest': 'Nottingham Forest FC',
        'Southampton': 'Southampton FC',
        'Spurs': 'Tottenham Hotspur FC',
        'West Ham': 'West Ham United FC',
        'Wolves': 'Wolverhampton Wanderers FC'
    }
    
    df['name'] = df['name'].map(name_map).fillna(df['name'])
    return df

def main():
    try:
        teams, players = fetch_fpl_data()
        processed_df = process_fpl_data(teams, players)
        final_df = normalize_names(processed_df)
        
        output_path = "data/raw/fpl_baselines_2024.csv"
        os.makedirs("data/raw", exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        print("\n--- FPL Baseline Data Created ---")
        print(final_df[['name', 'fpl_strength_home', 'fan_ownership_score']].sort_values(by='fan_ownership_score', ascending=False).head())
        print(f"\nSaved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
