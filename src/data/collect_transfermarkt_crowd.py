import requests
import pandas as pd
import os

def main():
    os.makedirs('data/raw', exist_ok=True)
    
    # Use GitHub mirror instead of blocked football-data.co.uk
    url = "https://raw.githubusercontent.com/k-eunji/epl202425_top5/main/E0.csv"
    print(f"Downloading odds data from mirror: {url}...")
    
    try:
        df = pd.read_csv(url)
        print("Download successful.")
        
        # Process data
        predictions = []
        for _, row in df.iterrows():
            try:
                # Basic validation
                if pd.isna(row['HomeTeam']) or pd.isna(row['B365H']):
                    continue
                    
                home_odd = float(row['B365H'])
                draw_odd = float(row['B365D'])
                away_odd = float(row['B365A'])
                
                # Determine crowd prediction (lowest odd wins)
                min_odd = min(home_odd, draw_odd, away_odd)
                
                if min_odd == home_odd:
                    pred = 'HOME_TEAM'
                elif min_odd == away_odd:
                    pred = 'AWAY_TEAM'
                else:
                    pred = 'DRAW'
                
                predictions.append({
                    'Date': row['Date'],
                    'HomeTeam': row['HomeTeam'],
                    'AwayTeam': row['AwayTeam'],
                    'HomeOdd': home_odd,
                    'DrawOdd': draw_odd,
                    'AwayOdd': away_odd,
                    'CrowdPrediction': pred,
                    'Source': 'Bet365 (via GitHub Mirror)'
                })
            except Exception as e:
                continue
                
        if predictions:
            df_out = pd.DataFrame(predictions)
            output_path = 'data/raw/crowd_predictions_2024.csv'
            df_out.to_csv(output_path, index=False)
            print(f"Saved {len(df_out)} crowd predictions to {output_path}")
        else:
            print("No valid data rows found.")

    except Exception as e:
        print(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    main()
