import json
import os
import requests
import time

def download_logos():
    os.makedirs('app/assets/logos', exist_ok=True)
    with open('data/processed/latest_team_stats.json', 'r') as f:
        data = json.load(f)
    teams = sorted(list(data['elo'].keys()))
    
    for team in teams:
        path = f"app/assets/logos/{team.replace(' ', '_')}.png"
        if os.path.exists(path):
            print(f"Skipping {team}, already exists.")
            continue
            
        print(f"Searching for {team}...")
        search_term = team.replace(' FC', '').replace(' AFC', '').replace(' & Hove Albion', '').replace(' Hotspur', '')
        
        try:
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={search_term}"
            res = requests.get(url, timeout=10)
            json_data = res.json()
            
            if json_data.get('teams'):
                # find the first soccer team from England
                valid_team = next((t for t in json_data['teams'] if t['strSport'] == 'Soccer' and t['strCountry'] == 'England'), None)
                if not valid_team:
                    valid_team = json_data['teams'][0] # fallback
                    
                badge_url = valid_team.get('strBadge')
                if badge_url:
                    img_res = requests.get(badge_url, timeout=10)
                    if img_res.status_code == 200:
                        with open(path, 'wb') as img_f:
                            img_f.write(img_res.content)
                        print(f"Downloaded {team} from {badge_url}")
                    else:
                        print(f"Failed to download image for {team}")
                else:
                    print(f"No badge URL for {team}")
            else:
                print(f"COULD NOT FIND TEAM: {team}")
                
        except Exception as e:
            print(f"Error for {team}: {e}")
            
        time.sleep(1) # Be nice to the free API

if __name__ == '__main__':
    download_logos()
