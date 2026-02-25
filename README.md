---
title: PremierPredict AI
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: app/dashboard.py
pinned: false
---

# PremierPredict-AI ⚽

**PremierPredict-AI** is an advanced Match Analysis Dashboard for Premier League matches, powered by a sophisticated Machine Learning Stacking Ensemble system.

## 🌟 Key Features

The dashboard recently underwent a massive V2 UI overhaul (Stitch Design) to provide a premium, data-rich experience:

- **AI Prediction Engine**: Combines XGBoost, RandomForest, GradientBoost, and a Logistic Regression Meta-Model to generate highly calibrated Win/Draw/Loss probabilities. 
- **Head-to-Head Analysis**: Visualizes team matchups dynamically, comparing Market Value, ELO Ratings, and Form with responsive team logos and custom dark-theme styling.
- **Comparative Feature Analysis**: Identifies the top statistical metrics (e.g., Attack Strength, Defensive Form) influencing the AI's current match prediction.
- **AI vs Human Performance Benchmark**: Tracks the AI's historical accuracy against baselines like Human Fan Confidence (FPL data) and BBC Expert predictions.
- **Native Dark Mode UI**: A fully custom, sleek dark design (`#1A202C`) utilizing Streamlit's native config controls alongside bulletproof CSS overrides for a seamless experience.

## 🛠 Tech Stack
- **Frontend**: Streamlit (Python) with custom injected HTML/CSS for advanced layouts.
- **Machine Learning**: `scikit-learn`, `xgboost` (Stacking Classifier).
- **Data Handling**: `pandas`, `joblib`.
- **Deployment**: Hugging Face Spaces.

## 🚀 How to Run Locally

1. Clone the repository: `git clone https://github.com/winnawat2023/PremierPredict-AI`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app/dashboard.py`

## 🧠 Model Architecture

The core of PremierPredict-AI is a **Stacking Ensemble System** (V5.0):
1. **Base Learners**: Diverse gradient boosting and tree-based models capture complex non-linear patterns in team statistics.
2. **Meta-Learner**: A Logistic Regression model dynamically weighs the confidence of the base learners to produce the final, highly calibrated match probability.

## 🔗 Live Application
This app is hosted live on Hugging Face Spaces.
