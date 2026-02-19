"""
Dixon-Coles Model for Football Match Prediction (Vectorized)
Based on: Dixon & Coles (1997) - "Modelling Association Football Scores"

The model uses Poisson distribution to estimate:
  - Attack strength (α) per team
  - Defense strength (β) per team  
  - Home advantage factor (γ)
  - Low-score correlation (ρ) for draw correction

Output: P(Home Win), P(Draw), P(Away Win) for each match
These are used as features in the ensemble model.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')


def dc_tau(x, y, lambda_home, lambda_away, rho):
    """Vectorized Dixon-Coles correction factor (τ)."""
    tau = np.ones_like(x, dtype=float)
    
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)
    
    tau[mask_00] = 1 - lambda_home[mask_00] * lambda_away[mask_00] * rho
    tau[mask_01] = 1 + lambda_home[mask_01] * rho
    tau[mask_10] = 1 + lambda_away[mask_10] * rho
    tau[mask_11] = 1 - rho
    
    return tau


def dc_log_like_vec(params, home_idx, away_idx, home_goals, away_goals, n_teams):
    """
    Vectorized negative log-likelihood for Dixon-Coles model.
    """
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    rho = params[2*n_teams + 1]
    
    # Expected goals (vectorized)
    lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    lambda_away = np.exp(attack[away_idx] + defense[home_idx])
    
    # Clip to prevent numerical issues
    lambda_home = np.clip(lambda_home, 0.001, 10.0)
    lambda_away = np.clip(lambda_away, 0.001, 10.0)
    
    # Poisson log-probabilities (vectorized)
    log_p_home = poisson.logpmf(home_goals, lambda_home)
    log_p_away = poisson.logpmf(away_goals, lambda_away)
    
    # Dixon-Coles correction
    tau = dc_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
    tau = np.clip(tau, 0.001, None)  # Avoid log(0)
    
    log_like = np.sum(log_p_home + log_p_away + np.log(tau))
    
    return -log_like


def predict_match(attack, defense, home_adv, rho, home_idx, away_idx, max_goals=8):
    """Predict match outcome probabilities using fitted parameters."""
    lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    lambda_away = np.exp(attack[away_idx] + defense[home_idx])
    
    lambda_home = np.clip(lambda_home, 0.001, 10.0)
    lambda_away = np.clip(lambda_away, 0.001, 10.0)
    
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            
            # Apply tau correction for low scores
            if i <= 1 and j <= 1:
                if i == 0 and j == 0:
                    p *= max(1 - lambda_home * lambda_away * rho, 0.001)
                elif i == 0 and j == 1:
                    p *= max(1 + lambda_home * rho, 0.001)
                elif i == 1 and j == 0:
                    p *= max(1 + lambda_away * rho, 0.001)
                elif i == 1 and j == 1:
                    p *= max(1 - rho, 0.001)
            
            if i > j:
                p_home_win += p
            elif i == j:
                p_draw += p
            else:
                p_away_win += p
    
    # Normalize
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        return p_home_win / total, p_draw / total, p_away_win / total
    return 0.4, 0.25, 0.35


def fit_dixon_coles(train_matches):
    """Fit the Dixon-Coles model using vectorized MLE."""
    # Get unique teams
    all_teams = sorted(set(train_matches['home_team'].tolist() + 
                          train_matches['away_team'].tolist()))
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    n_teams = len(all_teams)
    
    print(f"  Fitting Dixon-Coles on {len(train_matches)} matches, {n_teams} teams...")
    
    # Pre-compute index arrays (vectorized)
    home_idx = train_matches['home_team'].map(team_to_idx).values
    away_idx = train_matches['away_team'].map(team_to_idx).values
    home_goals = train_matches['home_score'].values.astype(int)
    away_goals = train_matches['away_score'].values.astype(int)
    
    # Initial parameters
    init_params = np.zeros(2 * n_teams + 2)
    init_params[2 * n_teams] = 0.25  # Home advantage
    init_params[2 * n_teams + 1] = -0.05  # Rho
    
    # Constraint: sum of attack strengths = 0
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_teams])}]
    
    # Bounds
    bounds = [(-3, 3)] * (2 * n_teams)  # Attack/Defense bounded
    bounds.append((-1, 1))  # Home advantage
    bounds.append((-0.99, 0.99))  # Rho
    
    result = minimize(
        dc_log_like_vec,
        init_params,
        args=(home_idx, away_idx, home_goals, away_goals, n_teams),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 200, 'disp': False, 'ftol': 1e-6}
    )
    
    if result.success:
        print(f"  ✅ Converged! (iterations: {result.nit})")
    else:
        print(f"  ⚠️  {result.message} (iterations: {result.nit})")
    
    attack = result.x[:n_teams]
    defense = result.x[n_teams:2*n_teams]
    home_adv = result.x[2*n_teams]
    rho = result.x[2*n_teams + 1]
    
    print(f"  Home advantage (γ): {home_adv:.4f}")
    print(f"  Draw correction (ρ): {rho:.4f}")
    
    # Show strongest teams
    team_stats = pd.DataFrame({
        'team': list(team_to_idx.keys()),
        'attack': [attack[team_to_idx[t]] for t in team_to_idx],
        'defense': [defense[team_to_idx[t]] for t in team_to_idx],
    })
    team_stats['overall'] = team_stats['attack'] - team_stats['defense']
    team_stats = team_stats.sort_values('overall', ascending=False)
    
    print(f"\n  Top 5 teams:")
    for _, row in team_stats.head(5).iterrows():
        print(f"    {row['team']:35s} Atk={row['attack']:+.3f} Def={row['defense']:+.3f}")
    print(f"  Bottom 3 teams:")
    for _, row in team_stats.tail(3).iterrows():
        print(f"    {row['team']:35s} Atk={row['attack']:+.3f} Def={row['defense']:+.3f}")
    
    return attack, defense, home_adv, rho, team_to_idx


def generate_dc_predictions(matches_df, attack, defense, home_adv, rho, team_to_idx):
    """Generate Dixon-Coles probabilities for each match."""
    results = []
    
    for _, row in matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        home_idx = team_to_idx.get(home)
        away_idx = team_to_idx.get(away)
        
        if home_idx is not None and away_idx is not None:
            ph, pd_, pa = predict_match(attack, defense, home_adv, rho, home_idx, away_idx)
        else:
            ph, pd_, pa = 0.40, 0.25, 0.35
        
        results.append({
            'dc_prob_home': ph,
            'dc_prob_draw': pd_,
            'dc_prob_away': pa
        })
    
    return pd.DataFrame(results, index=matches_df.index)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("DIXON-COLES MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/raw/all_matches_2020_2024.csv')
    df = df.dropna(subset=['home_score', 'away_score', 'winner'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    train = df[df['season'] != 2024]
    test = df[df['season'] == 2024]
    
    print(f"\nTrain: {len(train)} | Test: {len(test)}")
    
    # Fit
    attack, defense, home_adv, rho, team_to_idx = fit_dixon_coles(train)
    
    # Predict
    print(f"\n--- Testing on {len(test)} matches ---")
    dc_preds = generate_dc_predictions(test, attack, defense, home_adv, rho, team_to_idx)
    
    # Evaluate
    pred_outcomes = dc_preds[['dc_prob_home', 'dc_prob_draw', 'dc_prob_away']].values.argmax(axis=1)
    label_map = {'HOME_TEAM': 0, 'DRAW': 1, 'AWAY_TEAM': 2}
    actual = test['winner'].map(label_map).values
    
    acc = accuracy_score(actual, pred_outcomes)
    print(f"\n{'='*60}")
    print(f"Dixon-Coles Standalone Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")
    
    # Show sample predictions
    test_sample = test.head(5).copy()
    dc_sample = dc_preds.head(5)
    for i, (_, row) in enumerate(test_sample.iterrows()):
        dc = dc_sample.iloc[i]
        print(f"  {row['home_team']:30s} vs {row['away_team']:30s} | "
              f"H:{dc['dc_prob_home']:.2f} D:{dc['dc_prob_draw']:.2f} A:{dc['dc_prob_away']:.2f} | "
              f"Actual: {row['winner']}")
