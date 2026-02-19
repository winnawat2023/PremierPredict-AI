"""
NLP Text Features for Match Prediction (Inspired by Beal et al., AAAI 2021)

Extracts text-based features from Guardian EPL articles:
  1. Load Guardian articles for 2024/25 season
  2. For each match, find articles from the 7 days before
  3. Extract sentences mentioning each team
  4. Compute sentiment & TF-IDF features per team context
  5. Output features for ensemble integration

Simplified approach vs paper:
  - Paper used OpenIE relation extraction → we use simpler keyword matching
  - Paper used Count Vectorizer → we use TF-IDF (better for varying doc lengths)
  - We add sentiment polarity as additional signal
"""

import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Team name aliases for text matching
TEAM_ALIASES = {
    'Arsenal FC': ['arsenal', 'gunners'],
    'Aston Villa FC': ['aston villa', 'villa'],
    'AFC Bournemouth': ['bournemouth', 'cherries'],
    'Brentford FC': ['brentford', 'bees'],
    'Brighton & Hove Albion FC': ['brighton', 'seagulls', 'albion'],
    'Chelsea FC': ['chelsea', 'blues'],
    'Crystal Palace FC': ['crystal palace', 'palace', 'eagles'],
    'Everton FC': ['everton', 'toffees'],
    'Fulham FC': ['fulham', 'cottagers'],
    'Ipswich Town FC': ['ipswich'],
    'Leicester City FC': ['leicester', 'foxes'],
    'Liverpool FC': ['liverpool', 'reds'],
    'Manchester City FC': ['manchester city', 'man city', 'city'],
    'Manchester United FC': ['manchester united', 'man utd', 'man united', 'united'],
    'Newcastle United FC': ['newcastle', 'magpies', 'toon'],
    'Nottingham Forest FC': ['nottingham forest', 'forest'],
    'Southampton FC': ['southampton', 'saints'],
    'Tottenham Hotspur FC': ['tottenham', 'spurs'],
    'West Ham United FC': ['west ham', 'hammers'],
    'Wolverhampton Wanderers FC': ['wolves', 'wolverhampton'],
}

# Simple football sentiment lexicon
POSITIVE_WORDS = {
    'win', 'won', 'victory', 'brilliant', 'excellent', 'dominant', 'impressive',
    'clinical', 'superb', 'outstanding', 'strong', 'confident', 'momentum',
    'unbeaten', 'title', 'champion', 'top', 'best', 'strengthen', 'boosted',
    'comeback', 'surge', 'charge', 'flying', 'inspired', 'lethal', 'devastating'
}

NEGATIVE_WORDS = {
    'loss', 'lost', 'defeat', 'poor', 'terrible', 'struggle', 'weak',
    'injury', 'injured', 'suspended', 'crisis', 'sacked', 'fired',
    'relegation', 'relegated', 'bottom', 'worst', 'slump', 'frustrated',
    'concern', 'doubt', 'uncertain', 'winless', 'miss', 'absent', 'ban'
}


def load_articles():
    """Load Guardian articles."""
    path = 'data/raw/guardian_epl_articles_2024.json'
    with open(path, 'r') as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")
    return articles


def extract_team_context(text, team_aliases):
    """
    Extract sentences mentioning a specific team.
    Returns the combined text context for that team.
    """
    sentences = re.split(r'[.!?]+', text.lower())
    team_sentences = []
    
    for sentence in sentences:
        for alias in team_aliases:
            if alias in sentence:
                team_sentences.append(sentence.strip())
                break
    
    return ' '.join(team_sentences)


def compute_sentiment(text):
    """Simple sentiment score based on football lexicon."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total  # Range: -1 to +1


def build_nlp_features(features_df, articles, window_days=7):
    """
    Build NLP features for each match in features_df.
    
    For each match:
    1. Find articles published in the `window_days` before the match
    2. Extract text context for home and away teams
    3. Compute: sentiment, text length, mention count
    4. Apply TF-IDF + SVD for text vector features
    """
    print(f"\n--- Building NLP Features (window={window_days} days) ---")
    
    # Parse article dates
    for a in articles:
        a['parsed_date'] = datetime.strptime(a['date'], '%Y-%m-%d')
    
    # We need match dates from the raw data
    raw_df = pd.read_csv('data/raw/all_matches_2020_2024.csv')
    raw_df = raw_df.dropna(subset=['home_score', 'away_score', 'winner'])
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    
    # Only process 2024 season matches (where we have articles)
    test_matches = raw_df[raw_df['season'] == 2024].copy()
    test_matches = test_matches.sort_values('date')
    
    print(f"Processing {len(test_matches)} matches (Season 2024)")
    
    # For each match, collect NLP features
    nlp_features = []
    home_texts_all = []
    away_texts_all = []
    
    for _, match in test_matches.iterrows():
        match_date = match['date']
        home_team = match['home_team']
        away_team = match['away_team']
        
        # Find articles in window before match
        window_start = match_date - timedelta(days=window_days)
        relevant_articles = [
            a for a in articles
            if window_start <= a['parsed_date'] <= match_date
        ]
        
        # Get team aliases
        home_aliases = TEAM_ALIASES.get(home_team, [home_team.lower().split(' ')[0]])
        away_aliases = TEAM_ALIASES.get(away_team, [away_team.lower().split(' ')[0]])
        
        # Extract text context for each team
        home_text = ''
        away_text = ''
        home_mentions = 0
        away_mentions = 0
        
        for article in relevant_articles:
            body = article.get('body', '') + ' ' + article.get('title', '')
            
            h_ctx = extract_team_context(body, home_aliases)
            a_ctx = extract_team_context(body, away_aliases)
            
            if h_ctx:
                home_text += ' ' + h_ctx
                home_mentions += 1
            if a_ctx:
                away_text += ' ' + a_ctx
                away_mentions += 1
        
        # Compute simple features
        home_sentiment = compute_sentiment(home_text)
        away_sentiment = compute_sentiment(away_text)
        
        nlp_features.append({
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date,
            'n_articles': len(relevant_articles),
            'home_mentions': home_mentions,
            'away_mentions': away_mentions,
            'home_sentiment': home_sentiment,
            'away_sentiment': away_sentiment,
            'sentiment_diff': home_sentiment - away_sentiment,
            'home_text_len': len(home_text),
            'away_text_len': len(away_text),
            'mention_ratio': home_mentions / max(away_mentions, 1),
        })
        
        home_texts_all.append(home_text if home_text else 'no context')
        away_texts_all.append(away_text if away_text else 'no context')
    
    nlp_df = pd.DataFrame(nlp_features)
    
    # TF-IDF + SVD for text vector features
    print("Computing TF-IDF vectors...")
    all_texts = home_texts_all + away_texts_all
    
    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    # Reduce dimensionality with SVD (like the paper's approach)
    n_components = 5
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_vectors = svd.fit_transform(tfidf_matrix)
    
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Split back into home and away
    n_matches = len(test_matches)
    home_vectors = text_vectors[:n_matches]
    away_vectors = text_vectors[n_matches:]
    
    # Add text vector features
    for i in range(n_components):
        nlp_df[f'home_text_v{i}'] = home_vectors[:, i]
        nlp_df[f'away_text_v{i}'] = away_vectors[:, i]
        nlp_df[f'text_diff_v{i}'] = home_vectors[:, i] - away_vectors[:, i]
    
    # Summary stats
    print(f"\nNLP Feature Summary:")
    print(f"  Avg articles per match window: {nlp_df['n_articles'].mean():.1f}")
    print(f"  Avg home mentions: {nlp_df['home_mentions'].mean():.1f}")
    print(f"  Avg away mentions: {nlp_df['away_mentions'].mean():.1f}")
    print(f"  Avg home sentiment: {nlp_df['home_sentiment'].mean():.3f}")
    print(f"  Avg away sentiment: {nlp_df['away_sentiment'].mean():.3f}")
    
    return nlp_df, test_matches


if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("NLP TEXT FEATURES FOR MATCH PREDICTION")
    print("=" * 60)
    
    articles = load_articles()
    features_df = pd.read_csv('data/processed/features.csv')
    
    nlp_df, test_matches = build_nlp_features(features_df, articles)
    
    # Merge NLP features with existing features
    test_feat = features_df[features_df['season'] == 2024].copy()
    
    # Create match key for joining
    test_feat['match_key'] = test_feat['home_team'] + ' vs ' + test_feat['away_team']
    nlp_df['match_key'] = nlp_df['home_team'] + ' vs ' + nlp_df['away_team']
    
    test_merged = test_feat.merge(
        nlp_df.drop(columns=['home_team', 'away_team', 'date']),
        on='match_key', how='left'
    )
    
    base_features = [
        'Home_Form_L5', 'Away_Form_L5',
        'Home_Avg_GF_L5', 'Home_Avg_GA_L5',
        'Away_Avg_GF_L5', 'Away_Avg_GA_L5',
        'Home_MV', 'Away_MV', 'MV_Diff',
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'H2H_Home_Wins', 'H2H_Away_Wins'
    ]
    
    nlp_simple_features = [
        'home_sentiment', 'away_sentiment', 'sentiment_diff',
        'home_mentions', 'away_mentions', 'mention_ratio',
    ]
    
    nlp_vector_features = [f'home_text_v{i}' for i in range(5)] + \
                          [f'away_text_v{i}' for i in range(5)] + \
                          [f'text_diff_v{i}' for i in range(5)]
    
    # Train on pre-2024, test on 2024
    train_df = features_df[features_df['season'] != 2024].dropna(subset=base_features)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (Cross-Validation on 2024 test set)")
    print(f"{'='*60}")
    
    from sklearn.model_selection import cross_val_score
    
    configs = [
        ("Base AI (14 feat)", base_features),
        ("Base + Sentiment (17 feat)", base_features + nlp_simple_features),
        ("Base + Text Vectors (29 feat)", base_features + nlp_vector_features),
        ("Base + All NLP (32 feat)", base_features + nlp_simple_features + nlp_vector_features),
    ]
    
    for name, feats in configs:
        X = test_merged[feats].dropna()
        y = test_merged.loc[X.index, 'target']
        
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.1, random_state=42
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"  {name}: {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")
    
    # Save NLP features for use in ensemble
    nlp_save_cols = ['match_key'] + nlp_simple_features + nlp_vector_features
    nlp_df[nlp_save_cols].to_csv('data/processed/nlp_features_2024.csv', index=False)
    print(f"\nNLP features saved to data/processed/nlp_features_2024.csv")
