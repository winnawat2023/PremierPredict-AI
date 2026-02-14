import os
import requests
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
PL_COMPETITION_ID = 2021  # Premier League ID

if not API_KEY:
    raise ValueError("API Key not found. Please set FOOTBALL_DATA_API_KEY in .env file.")

HEADERS = {
    "X-Auth-Token": API_KEY
}

def fetch_season_data(season_year):
    """
    Fetch all matches for a specific PL season.
    season_year: Start year of the season (e.g., 2023 for 2023-2024 season)
    """
    url = f"{BASE_URL}/competitions/{PL_COMPETITION_ID}/matches"
    params = {
        "season": season_year
    }
    
    print(f"Fetching data for season {season_year}...")
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None
        
    return response.json()

def parse_matches(data):
    """
    Parse JSON response into a list of dictionaries for DataFrame.
    """
    matches = []
    if not data or 'matches' not in data:
        return matches
        
    for match in data['matches']:
        match_info = {
            'id': match['id'],
            'date': match['utcDate'],
            'season': match['season']['startDate'][:4],
            'home_team': match['homeTeam']['name'],
            'away_team': match['awayTeam']['name'],
            'home_score': match['score']['fullTime']['home'],
            'away_score': match['score']['fullTime']['away'],
            'winner': match['score']['winner'],  # HOME_TEAM, AWAY_TEAM, DRAW
            'status': match['status']
        }
        matches.append(match_info)
    return matches

def main():
    # Seasons 2021, 2022, 2023, 2024
    seasons = [2021, 2022, 2023, 2024, 2025]
    all_matches = []
    
    for season in seasons:
        data = fetch_season_data(season)
        season_matches = parse_matches(data)
        all_matches.extend(season_matches)
        
    if all_matches:
        df = pd.DataFrame(all_matches)
        
        # Ensure output directory exists
        os.makedirs("data/raw", exist_ok=True)
        
        output_path = "data/raw/pl_matches_2021_2025.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} matches to {output_path}")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
