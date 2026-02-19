import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
import random

# Output file
OUTPUT_FILE = 'data/raw/xg_stats.csv'

# Seasons to scrape (2020-2024)
# Understat uses starting year (e.g. 2023 for 23/24)
SEASONS = [2020, 2021, 2022, 2023, 2024]

BASE_URL = "https://understat.com/league/EPL/{}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def decode_hex_string(s):
    """
    Decode a string that contains hex escapes like \x7B for {
    """
    return s.encode('utf-8').decode('unicode_escape')

def scrape_xg_stats():
    all_data = []

    for season in SEASONS:
        url = BASE_URL.format(season)
        print(f"Scraping Season {season} from {url}...")
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch {season}: Status {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            teams_data = None
            
            # Iterate through all scripts to find the one with team data
            for script in scripts:
                if not script.string:
                    continue
                
                # Look for all JSON.parse calls
                # Pattern: JSON.parse('...')
                # We want the content inside the quotes
                matches = re.finditer(r"JSON\.parse\('([^']+)'\)", script.string)
                
                for match in matches:
                    json_str = match.group(1)
                    try:
                        decoded_json_str = decode_hex_string(json_str)
                        data = json.loads(decoded_json_str)
                        
                        # Check if this data looks like teamsData
                        # Key features: dictionary where keys are IDs, values have 'id', 'title', 'history'
                        if isinstance(data, dict) and len(data) > 0:
                            first_key = list(data.keys())[0]
                            first_val = data[first_key]
                            if isinstance(first_val, dict) and 'title' in first_val and 'history' in first_val:
                                print("  Found valid teamsData!")
                                teams_data = data
                                break
                    except Exception as e:
                        # Not a valid JSON or decode error, ignore
                        pass
                
                if teams_data:
                    break
            
            if not teams_data:
                print(f"Could not find teamsData for {season}")
                # Debug: print first 500 chars of scripts to see what's there
                # for s in scripts:
                #    if s.string and "JSON.parse" in s.string:
                #        print("Chunk:", s.string[:200])
                continue
            
            # Parse teamsData
            for team_id, team_info in teams_data.items():
                team_name = team_info['title']
                history = team_info['history']
                
                total_xg = 0.0
                total_xga = 0.0
                matches_played = len(history)
                
                for match in history:
                    total_xg += float(match['xG'])
                    total_xga += float(match['xGA'])
                
                if matches_played > 0:
                    avg_xg = total_xg / matches_played
                    avg_xga = total_xga / matches_played
                else:
                    avg_xg = 0.0
                    avg_xga = 0.0
                
                all_data.append({
                    'season': season,
                    'team_name': team_name,
                    'matches_played': matches_played,
                    'total_xg': total_xg,
                    'total_xga': total_xga,
                    'avg_xg': avg_xg,
                    'avg_xga': avg_xga
                })
                    
            print(f"  Collected {len(teams_data)} teams.")
            
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error scraping {season}: {e}")
            
    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved {len(df)} records to {OUTPUT_FILE}")
        print(df.head())
    else:
        print("No data collected.")

if __name__ == "__main__":
    scrape_xg_stats()
