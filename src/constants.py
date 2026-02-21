"""
Centralized constants for PremierPredict-AI
"""

# Standard team names (Football-Data API names)
TEAM_NAMES = [
    'Arsenal FC', 'Aston Villa FC', 'AFC Bournemouth', 'Brentford FC',
    'Brighton & Hove Albion FC', 'Chelsea FC', 'Crystal Palace FC',
    'Everton FC', 'Fulham FC', 'Ipswich Town FC', 'Leicester City FC',
    'Liverpool FC', 'Manchester City FC', 'Manchester United FC',
    'Newcastle United FC', 'Nottingham Forest FC', 'Southampton FC',
    'Tottenham Hotspur FC', 'West Ham United FC', 'Wolverhampton Wanderers FC',
    'Luton Town FC', 'Burnley FC', 'Sheffield United FC', 'Leeds United FC'
]

# Mapping from various sources (Betting, FPL, Guardian) to standard names
TEAM_NAME_MAPPING = {
    # Betting/Crowd Data aliases
    'Man United': 'Manchester United FC',
    'Man Utd': 'Manchester United FC',
    'Manchester Utd': 'Manchester United FC',
    'Man City': 'Manchester City FC',
    'Liverpool': 'Liverpool FC',
    'Arsenal': 'Arsenal FC',
    'Aston Villa': 'Aston Villa FC',
    'Tottenham': 'Tottenham Hotspur FC',
    'Spurs': 'Tottenham Hotspur FC',
    'Chelsea': 'Chelsea FC',
    'Newcastle': 'Newcastle United FC',
    'West Ham': 'West Ham United FC',
    'Brighton': 'Brighton & Hove Albion FC',
    'Brentford': 'Brentford FC',
    'Crystal Palace': 'Crystal Palace FC',
    'Wolves': 'Wolverhampton Wanderers FC',
    'Fulham': 'Fulham FC',
    'Bournemouth': 'AFC Bournemouth',
    'Everton': 'Everton FC',
    "Nott'm Forest": 'Nottingham Forest FC',
    'Nottingham Forest': 'Nottingham Forest FC',
    'Luton': 'Luton Town FC',
    'Burnley': 'Burnley FC',
    'Sheffield United': 'Sheffield United FC',
    'Sheffield Utd': 'Sheffield United FC',
    'Leicester': 'Leicester City FC',
    'Leicester City': 'Leicester City FC',
    'Ipswich': 'Ipswich Town FC',
    'Ipswich Town': 'Ipswich Town FC',
    'Southampton': 'Southampton FC',
    'Leeds': 'Leeds United FC',
    'Leeds United': 'Leeds United FC',
    
    # NLP/Guardian aliases (lowercase for search)
    'gunners': 'Arsenal FC',
    'cherries': 'AFC Bournemouth',
    'bees': 'Brentford FC',
    'seagulls': 'Brighton & Hove Albion FC',
    'blues': 'Chelsea FC',
    'eagles': 'Crystal Palace FC',
    'toffees': 'Everton FC',
    'cottagers': 'Fulham FC',
    'tractor boys': 'Ipswich Town FC',
    'foxes': 'Leicester City FC',
    'reds': 'Liverpool FC',
    'citizens': 'Manchester City FC',
    'red devils': 'Manchester United FC',
    'magpies': 'Newcastle United FC',
    'forest': 'Nottingham Forest FC',
    'saints': 'Southampton FC',
    'lilywhites': 'Tottenham Hotspur FC',
    'hammers': 'West Ham United FC'
}

# Features used in models
BASE_FEATURES = [
    'Home_Form_L5', 'Away_Form_L5',
    'Home_Avg_GF_L5', 'Home_Avg_GA_L5',
    'Away_Avg_GF_L5', 'Away_Avg_GA_L5',
    'Home_MV', 'Away_MV', 'MV_Diff',
    'Home_Elo', 'Away_Elo', 'Elo_Diff',
    'H2H_Home_Wins', 'H2H_Away_Wins'
]

HYBRID_FEATURES = BASE_FEATURES + [
    'fpl_home', 'fpl_away', 'fpl_diff',
    'elo_prob_home', 'elo_prob_draw', 'elo_prob_away'
]
