import requests
from bs4 import BeautifulSoup
import json
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def test_transfermarkt():
    print("Testing Transfermarkt...")
    url = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id=2023"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for the table with market values
            # Specifically looking for 'items' table
            table = soup.find('table', class_='items')
            if table:
                print("Found 'items' table.")
                # Print first row to verify
                rows = table.find_all('tr', class_=['odd', 'even'])
                if rows:
                    print(f"Found {len(rows)} rows.")
                    first_row = rows[0]
                    # usually team name is in td class="hauptlink"
                    team_name_td = first_row.find('td', class_='hauptlink')
                    if team_name_td:
                        print(f"First Team: {team_name_td.text.strip()}")
            else:
                print("Could not find 'items' table. Structure might be different or blocked.")
    except Exception as e:
        print(f"Error scraping TM: {e}")

def test_understat():
    print("\nTesting Understat...")
    url = "https://understat.com/league/EPL/2023"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for scripts with JSON data
            scripts = soup.find_all('script')
            found_data = False
            for script in scripts:
                if script.string and "teamsData" in script.string:
                    print("Found 'teamsData' in script.")
                    found_data = True
                    break
            if not found_data:
                print("Could not find teamsData script.")
    except Exception as e:
        print(f"Error scraping Understat: {e}")

if __name__ == "__main__":
    test_transfermarkt()
    test_understat()
