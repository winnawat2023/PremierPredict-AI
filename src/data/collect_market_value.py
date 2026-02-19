import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# Output file
OUTPUT_FILE = 'data/raw/market_values.csv'

# Seasons to scrape (2020-2024)
# Note: Transfermarkt uses the starting year of the season (e.g., 2023 for 23/24)
SEASONS = [2020, 2021, 2022, 2023, 2024]

# Transfermarkt URL pattern for Premier League (GB1)
BASE_URL = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def clean_currency(value_str):
    """
    Convert currency string (e.g., '€1.23bn', '€500.00m') to float (in millions).
    Returns value in Millions (e.g. 1230.0 for 1.23bn).
    """
    if not isinstance(value_str, str):
        return 0.0
    
    # Remove currency symbol and whitespace
    value_str = value_str.replace('€', '').strip()
    
    try:
        if 'bn' in value_str:
            # Billions -> convert to Millions
            val = float(value_str.replace('bn', ''))
            return val * 1000
        elif 'm' in value_str:
            # Millions -> keep as is
            val = float(value_str.replace('m', ''))
            return val
        elif 'k' in value_str:
             # Thousands -> convert to Millions
            val = float(value_str.replace('k', ''))
            return val / 1000
        else:
            return 0.0
    except ValueError:
        return 0.0

def scrape_market_values():
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
            
            # Find the main table. Usually id='yw1' or class='items'
            table = soup.find('table', class_='items')
            if not table:
                print(f"Could not find data table for {season}")
                continue
                
            # Rows are in tbody, class 'odd' or 'even'
            tbody = table.find('tbody')
            if not tbody:
                 print(f"Could not find tbody for {season}")
                 continue
                 
            rows = tbody.find_all('tr')
            
            for row in rows:
                # Team Name is usually in a td with class 'hauptlink' (the one with the link)
                # Market Value is usually in the last columns, specifically class 'rechts hauptlink'
                
                cols = row.find_all('td')
                
                # Check for Team Name
                team_cell = row.find('td', class_='hauptlink')
                if not team_cell:
                     # Sometimes layout differs, look for a link inside
                     continue
                
                # The text might contain newlines, strip it
                team_name = team_cell.text.strip()
                if not team_name:
                    # fallback
                    a_tag = team_cell.find('a')
                    if a_tag:
                         team_name = a_tag.get('title', '').strip() or a_tag.text.strip()

                # Market Value
                # It's usually the last cell or the one with class 'rechts hauptlink'
                # Let's look for 'rechts hauptlink' first
                mv_cell = row.find('td', class_='rechts hauptlink')
                
                # If not found, try searching just 'rechts' at the end of row
                if not mv_cell:
                     # iterate backwards
                     for cell in reversed(cols):
                         if 'rechts' in cell.get('class', []) and '€' in cell.text:
                             mv_cell = cell
                             break
                
                market_value_str = "0"
                if mv_cell:
                    market_value_str = mv_cell.text.strip()
                    # extract just the text link if valid
                    a_tag = mv_cell.find('a')
                    if a_tag:
                        market_value_str = a_tag.text.strip()

                market_value_m = clean_currency(market_value_str)
                
                if team_name and market_value_m > 0:
                    all_data.append({
                        'season': season,
                        'club_name': team_name,
                        'market_value_m': market_value_m 
                    })
                    
            print(f"  Collected {len(all_data) - (len(all_data) - len(rows))} rows (approx). Total so far: {len(all_data)}")
            
            # Be polite
            time.sleep(random.uniform(2, 5))
            
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
    scrape_market_values()
