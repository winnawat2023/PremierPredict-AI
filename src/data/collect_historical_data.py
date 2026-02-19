import pandas as pd
import requests
import io
import os
from datetime import datetime

MATCHES_FILE = 'data/raw/pl_matches_2021_2025.csv' # Existing file (2023-2025)
OUTPUT_FILE = 'data/raw/all_matches_2020_2024.csv' # Combined file

# URLs for footballcsv and cache.footballdata
# 2020: footballcsv/england (Standard)
# 2021, 2022: footballcsv/cache.footballdata (Verified backup for betting odds/consistency)
URLS = {
    2020: "https://raw.githubusercontent.com/footballcsv/england/master/2020s/2020-21/eng.1.csv",
    2021: "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master/2021-22/eng.1.csv",
    2022: "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master/2022-23/eng.1.csv"
}

def normalize_team_name(name):
    """
    Normalize team names to match the format in standard dataset.
    footballcsv names are usually just 'Arsenal', 'Chelsea', etc.
    We map them to 'Arsenal FC', 'Chelsea FC' etc.
    """
    name_map = {
        "Man City": "Manchester City FC",
        "Man Utd": "Manchester United FC",
        "Manchester City": "Manchester City FC",
        "Manchester United": "Manchester United FC",
        "Liverpool": "Liverpool FC",
        "Arsenal": "Arsenal FC",
        "Aston Villa": "Aston Villa FC",
        "Tottenham": "Tottenham Hotspur FC",
        "Spurs": "Tottenham Hotspur FC",
        "Chelsea": "Chelsea FC",
        "Newcastle": "Newcastle United FC",
        "Newcastle Utd": "Newcastle United FC",
        "West Ham": "West Ham United FC",
        "Brighton": "Brighton & Hove Albion FC",
        "Brentford": "Brentford FC",
        "Crystal Palace": "Crystal Palace FC",
        "Wolves": "Wolverhampton Wanderers FC",
        "Fulham": "Fulham FC",
        "Bournemouth": "AFC Bournemouth",
        "Everton": "Everton FC",
        "Nott'm Forest": "Nottingham Forest FC",
        "Nottingham Forest": "Nottingham Forest FC",
        "Luton": "Luton Town FC",
        "Burnley": "Burnley FC",
        "Sheffield United": "Sheffield United FC",
        "Sheffield Utd": "Sheffield United FC",
        "Leicester": "Leicester City FC",
        "Leicester City": "Leicester City FC",
        "Ipswich": "Ipswich Town FC",
        "Ipswich Town": "Ipswich Town FC",
        "Southampton": "Southampton FC",
        "Leeds": "Leeds United FC",
        "Leeds United": "Leeds United FC",
        "West Brom": "West Bromwich Albion FC",
        "Watford": "Watford FC",
        "Norwich": "Norwich City FC"
    }
    return name_map.get(name, name)

def fetch_and_process():
    all_dfs = []
    
    # 1. Fetch Historical Data
    for season, url in URLS.items():
        print(f"Fetching Season {season} from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Read CSV
            df = pd.read_csv(io.StringIO(response.text))
            
            # footballcsv format: Date, Team 1, FT, Team 2
            # FT is like "2-1" or "0–3" (en-dash)
            
            processed_rows = []
            for _, row in df.iterrows():
                try:
                    score = str(row['FT'])
                    
                    # Handle En-Dash and standard Hyphen
                    if '–' in score:
                         score = score.replace('–', '-')
                    
                    if '-' not in score:
                        continue
                        
                    home_goals, away_goals = map(int, score.split('-'))
                    
                    winner = 'DRAW'
                    if home_goals > away_goals:
                        winner = 'HOME_TEAM'
                    elif away_goals > home_goals:
                        winner = 'AWAY_TEAM'
                        
                    processed_rows.append({
                        'id': f"hist_{season}_{_}",
                        'date': pd.to_datetime(row['Date']),
                        'season': season,
                        'home_team': normalize_team_name(row['Team 1']),
                        'away_team': normalize_team_name(row['Team 2']),
                        'home_score': home_goals,
                        'away_score': away_goals,
                        'winner': winner,
                        'status': 'FINISHED'
                    })
                except Exception as e:
                    print(f"Skipping row in {season}: {row.values} - Error: {e}")

            all_dfs.append(pd.DataFrame(processed_rows))
            print(f"  Processed {len(processed_rows)} matches.")
            
        except Exception as e:
            print(f"Failed to fetch {season}: {e}")

    # 2. Load Existing Data
    if os.path.exists(MATCHES_FILE):
        print(f"Loading existing data from {MATCHES_FILE}...")
        df_existing = pd.read_csv(MATCHES_FILE)
        
        # Convert existing to tz-naive
        df_existing['date'] = pd.to_datetime(df_existing['date'], utc=True).dt.tz_localize(None)
        
        all_dfs.append(df_existing)
    else:
        print("Warning: Existing dataset not found.")

    # 3. Merge and Save
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.sort_values(by='date', inplace=True)
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved {len(final_df)} matches (2020-2025) to {OUTPUT_FILE}")
        
    else:
        print("No data collected.")

if __name__ == "__main__":
    fetch_and_process()
