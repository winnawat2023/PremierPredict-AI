import pandas as pd
import requests
from io import StringIO

def check_url(url, name):
    print(f"--- Checking {name} ---")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print(f"Success! Columns: {list(df.columns)[:15]}...")
            if 'selected_by_percent' in df.columns:
                 print("Found Ownership Data!")
                 print(df[['web_name', 'selected_by_percent', 'now_cost']].sort_values(by='selected_by_percent', ascending=False).head())
        else:
            print(f"Failed: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

check_url("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv", "Players Raw CSV")
