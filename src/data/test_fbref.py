import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def test_fbref():
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    print(f"Testing FBref: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers) # timeout removed for simple test
        print(f"Status: {response.status_code}")
        
        # FBRef often comments out tables to save bandwidth? No, that's for some parts.
        # Main table 'stats_squads_standard_for' usually has xG
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for table with id 'stats_squads_standard_for'
            table = soup.find('table', id='stats_squads_standard_for')
            if table:
                print("Found 'stats_squads_standard_for' table.")
                # Parse with pandas
                dfs = pd.read_html(str(table))
                if dfs:
                    df = dfs[0]
                    print("Columns:", df.columns)
                    # Check for xG column. FBref uses multi-level columns usually.
                    # It might be ('Expected', 'xG') or similar.
                    # Let's flatten columns to check
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                    print(df.head())
            else:
                print("Table 'stats_squads_standard_for' not found. Checking comments...")
                # checking for commented out tables
                from bs4 import Comment
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for c in comments:
                    if 'stats_squads_standard_for' in c:
                        print("Found table inside comments!")
                        break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fbref()
