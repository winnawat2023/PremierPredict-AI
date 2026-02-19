import requests
from bs4 import BeautifulSoup
from googlesearch import search
import pandas as pd
import re
import time
import os
from datetime import datetime

def normalize_team_name(name):
    """
    Normalize team names to match the format in pl_matches_2021_2025.csv
    This is a basic mapping and might need expansion.
    """
    name_map = {
        "Man City": "Manchester City FC",
        "Man Utd": "Manchester United FC",
        "Manchester United": "Manchester United FC",
        "Manchester City": "Manchester City FC",
        "Spurs": "Tottenham Hotspur FC",
        "Tottenham": "Tottenham Hotspur FC",
        "Arsenal": "Arsenal FC",
        "Aston Villa": "Aston Villa FC",
        "Bournemouth": "AFC Bournemouth",
        "Brentford": "Brentford FC",
        "Brighton": "Brighton & Hove Albion FC",
        "Burnley": "Burnley FC",
        "Chelsea": "Chelsea FC",
        "Crystal Palace": "Crystal Palace FC",
        "Everton": "Everton FC",
        "Fulham": "Fulham FC",
        "Ipswich": "Ipswich Town FC",
        "Leicester": "Leicester City FC",
        "Liverpool": "Liverpool FC",
        "Luton": "Luton Town FC",
        "Newcastle": "Newcastle United FC",
        "Nott'm Forest": "Nottingham Forest FC",
        "Nottingham Forest": "Nottingham Forest FC",
        "Forest": "Nottingham Forest FC",
        "Sheffield Utd": "Sheffield United FC",
        "Sheffield United": "Sheffield United FC",
        "Southampton": "Southampton FC",
        "West Ham": "West Ham United FC",
        "Wolves": "Wolverhampton Wanderers FC",
        "Leeds": "Leeds United FC"
    }
    return name_map.get(name, name)

def get_prediction_links(num_results=20):
    query = "Chris Sutton Premier League predictions site:bbc.com/sport"
    links = []
    print(f"Searching for articles with query: '{query}'...")
    try:
        # search returns a generator
        for result in search(query, num_results=num_results, advanced=True):
             # Filter for actual prediction articles, usually 2024 or 2025 in title/snippet ideally
             # But Google search advanced results object needs handling
             links.append(result.url)
    except Exception as e:
        print(f"Search failed: {e}")
    
    print(f"Found {len(links)} potential links.")
    return links

def parse_prediction_article(url):
    print(f"Parsing: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        predictions = []
        
        # BBC prediction articles often use specific headings for matches
        # e.g. "Arsenal v Manchester United" followed by text "Sutton's prediction: 2-1"
        
        # Pattern to find score predictions: "Sutton's prediction: X-Y"
        # It's often in a text block
        
        # We look for h2 or h3 tags that look like "Team A v Team B"
        match_headers = soup.find_all(['h2', 'h3'])
        
        for header in match_headers:
            header_text = header.get_text().strip()
            if ' v ' in header_text:
                teams = header_text.split(' v ')
                if len(teams) == 2:
                    home_team = normalize_team_name(teams[0].strip())
                    away_team = normalize_team_name(teams[1].strip())
                    
                    # Now look for the prediction in the siblings following this header
                    # until the next header
                    current_elem = header.next_sibling
                    sutton_score = None
                    
                    while current_elem and current_elem.name not in ['h2', 'h3']:
                        if hasattr(current_elem, 'get_text'):
                            text = current_elem.get_text()
                            # Regex for "Sutton's prediction: 1-2" or similar
                            # Sometimes it might be "Sutton's prediction: 1-2"
                            match = re.search(r"Sutton's prediction:\s*(\d+)-(\d+)", text, re.IGNORECASE)
                            if match:
                                sutton_score = (int(match.group(1)), int(match.group(2)))
                                break
                        current_elem = current_elem.next_sibling
                    
                    if sutton_score:
                        print(f"  Found: {home_team} vs {away_team} -> {sutton_score}")
                        predictions.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'sutton_home_score': sutton_score[0],
                            'sutton_away_score': sutton_score[1],
                            'source_url': url,
                            'scraped_at': datetime.now().isoformat()
                        })
        
        return predictions

    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

def main():
    all_predictions = []
    links = get_prediction_links(num_results=20) # Start with 20 to test
    
    unique_links = list(set(links)) # Deduplicate
    
    count = 0
    for link in unique_links:
        # Filter for likely prediction pages
        if "prediction" not in link.lower() and "sutton" not in link.lower():
             continue
             
        try:
            preds = parse_prediction_article(link)
            if preds:
                all_predictions.extend(preds)
            count += 1
            time.sleep(1) # Be polite
            if count >= 10: # Limit for first run
                break
        except Exception as e:
            print(f"Skipping {link}: {e}")

    if all_predictions:
        df = pd.DataFrame(all_predictions)
        output_path = 'data/raw/sutton_predictions_2024_2025.csv'
        os.makedirs('data/raw', exist_ok=True)
        
        # Check if file exists to appeal (or just overwrite for now as we are experimenting)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_predictions)} predictions to {output_path}")
    else:
        print("No predictions found.")

if __name__ == "__main__":
    main()
