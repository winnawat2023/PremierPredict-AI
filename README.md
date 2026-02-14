# PremierPredict-AI
**Subject**: SEA612 Artificial Intelligence Fundamentals (KMUTT - SIT)  
**Instructor**: Punnarust Silparattanawong  
**Owner**: Chaiyaporn Homthian (Bass)

## Project Goal
**Research Problem**: Match Outcome Prediction (Classification)  
**Goal**: Prove AI (Random Forest) accuracy > Baseline (Home Win Strategy).  
**Baseline**: Always predict Home Win.  
**AI Model**: Random Forest Classifier (handling complex statistical features).

## Technical Design
**Input Features (X)**:
1. `Home_Advantage`: Binary or calculated advantage.
2. `Form_L5`: Average points in last 5 matches.
3. `Position_Diff`: League table position difference.

**Target (Y)**:
- 0: Home Win
- 1: Draw
- 2: Away Win

## Milestones
- **22/02/2026**: Project Proposal (Research Problem & Design)
- **27/03/2026**: Final Report (IEEE Template 4 pages)
- **29/03/2026**: Project Presentation

## Data & Tech Stack
- **Source**: Football-Data.org API (English Premier League 2021-2025)
- **Stack**: Python, Scikit-learn, Pandas, Streamlit, Git

## Folder Structure
- `data/`: Raw and processed data CSVs
- `src/`: 
    - `data_loader.py`: Fetch data from API
    - `features.py`: Feature engineering logic
    - `models.py`: Training and comparison logic
- `app/`: Streamlit dashboard
- `main.py`: Entry point


