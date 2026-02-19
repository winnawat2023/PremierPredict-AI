import pandas as pd
import requests
from io import StringIO

def check_url(url, name):
    print(f"--- Checking {name} ---")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print(f"Success! Columns: {list(df.columns)[:10]}...")
            print(f"Shape: {df.shape}")
            if 'strength' in df.columns or 'strength_overall_home' in df.columns:
                print("Found Strength Ratings!")
                print(df[['name', 'strength_overall_home', 'strength_overall_away']].head())
            if 'team_h_difficulty' in df.columns:
                print("Found Difficulty Ratings in Fixtures!")
        else:
            print(f"Failed: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

check_url("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/teams.csv", "Teams CSV")
check_url("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/fixtures.csv", "Fixtures CSV")
