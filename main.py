import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    print("PremierPredict-AI: Project Initialized")
    print(f"API Key present: {bool(os.getenv('FOOTBALL_DATA_API_KEY'))}")

if __name__ == "__main__":
    main()
