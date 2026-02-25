import streamlit as st
import os
import sys
import pandas as pd
import joblib
import json
import altair as alt
import datetime
import base64

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set page config
st.set_page_config(page_title="PremierPredict-AI", layout="wide", page_icon="⚽", initial_sidebar_state="expanded")

# Custom CSS for the New Match Analysis Dashboard Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Base App Theme */
    html, body, .stApp {
        background-color: #11141E !important; /* Very dark blue background with subtle texture */
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Override Streamlit Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161A25 !important;
        border-right: 1px solid #1f2937 !important;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Header */
    header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 1400px; }
    
    .dash-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .dash-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 30px;
    }
    .status-dot {
        width: 8px; height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }
    
    /* Custom Styling for Cards */
    .card {
        background-color: #1A202C !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    .card-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    
    /* Data Bars and Texts */
    .cyan-text { color: #06b6d4; font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.2rem; }
    .red-text { color: #f43f5e; font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.2rem; }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    /* Progress lines */
    .line-home { height: 3px; background-color: #06b6d4; width: 60%; margin-top: 5px; border-radius: 2px;}
    .line-away { height: 3px; background-color: #f43f5e; width: 80%; margin-top: 5px; border-radius: 2px; margin-left: auto;}
    
    /* Sidebar specific */
    .sidebar-logo {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #3b82f6;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 5px;
    }
    .sidebar-section {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Selectbox Overrides for Sidebar */
    div[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: #1A202C !important;
        border-radius: 4px;
        border: 1px solid #2d3748 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
        background-color: transparent !important;
        color: #f8fafc !important;
    }
    div[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:hover {
        background-color: #2d3748 !important;
    }
    
    /* Active Models tags - exact solid colors */
    .model-tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 500;
        margin: 3px 2px;
        color: #e2e8f0;
    }
    
    /* Run Button */
    .stButton > button {
        background: #3b82f6 !important;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 0;
        width: 100%;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 2rem;
        transition: all 0.2s;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    
    /* Prediction Box */
    .pred-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, #1A202C 100%);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 15px 20px;
        position: relative;
        overflow: hidden;
    }
    .pred-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 6px;
        background-color: #10b981;
        box-shadow: 0 0 15px #10b981;
    }
    .pred-title {
        font-family: 'Orbitron', sans-serif;
        color: #f8fafc;
        font-size: 1.1rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pred-result-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.8rem;
        color: #e2e8f0;
        margin: 0;
        line-height: 1.2;
    }
    .confidence-badge {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 10px;
    }
    
    /* Influencing Features Box */
    .feature-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .impact-pos { color: #10b981; }
    .impact-neg { color: #f43f5e; }
    
    /* Data source tag */
    .data-source {
        font-size: 0.6rem;
        color: #64748b;
        background-color: #1e293b;
        padding: 4px 8px;
        border-radius: 12px;
        float: right;
    }
    
    /* Dropdown Target Wrappers */
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"]:first-of-type > div > div {
        border-bottom: 3px solid #06b6d4 !important;
        border-radius: 4px;
        box-shadow: none;
    }
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"]:last-of-type > div > div {
        border-bottom: 3px solid #f43f5e !important;
        border-radius: 4px;
        box-shadow: none;
    }
    
    /* Hide selectbox label */
    .stSelectbox label { display: none; }
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_PATH = "models/stacking_ensemble_v5.pkl"
STATS_PATH = "data/processed/latest_team_stats.json"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from src.utils import normalize_team_name

# Cache Compatibility
if hasattr(st, 'cache_resource'):
    cache_decorator = st.cache_resource
else:
    try:
        cache_decorator = st.experimental_singleton
    except AttributeError:
        cache_decorator = st.cache(allow_output_mutation=True)

@cache_decorator
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None, None
    try:
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)
    except FileNotFoundError:
        return None, None
    return model, stats

def main():
    model, stats = load_resources()
    if not model or not stats:
        st.error("Could not load backend models/stats.")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🤖 PremierPredict</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem; color:#64748b; margin-bottom:1rem;">AI ENSEMBLE SYSTEM V5.0</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">MATCH SIMULATION</div>', unsafe_allow_html=True)
        
        team_list = sorted(list(stats['elo'].keys()))
        app_dir = os.path.dirname(__file__)
        
        home_team_val = st.session_state.get("home", team_list[0] if team_list else "AFC Bournemouth")
        home_logo_name = normalize_team_name(home_team_val).replace(" ", "_")
        home_logo_path = os.path.join(app_dir, "assets", "logos", f"{home_logo_name}.png")
        home_logo_b64 = get_base64_image(home_logo_path)
        home_img_tag = f'<img src="data:image/png;base64,{home_logo_b64}" width="20" style="vertical-align: middle; margin-right: 8px;">' if home_logo_b64 else ''
        
        st.markdown(f'<div style="font-size:0.75rem; color:#06b6d4; font-weight:700; margin-bottom:5px; letter-spacing:1px; display:flex; align-items:center;">{home_img_tag}HOME TEAM</div>', unsafe_allow_html=True)
        home_team = st.selectbox("home", team_list, index=team_list.index(home_team_val) if home_team_val in team_list else 0, key="home")
        
        st.markdown('<div style="text-align:center; color:#64748b; margin:10px 0;">⚔️</div>', unsafe_allow_html=True)
        
        away_team_val = st.session_state.get("away", team_list[-1] if team_list else "Wolverhampton Wanderers FC")
        away_logo_name = normalize_team_name(away_team_val).replace(" ", "_")
        away_logo_path = os.path.join(app_dir, "assets", "logos", f"{away_logo_name}.png")
        away_logo_b64 = get_base64_image(away_logo_path)
        away_img_tag = f'<img src="data:image/png;base64,{away_logo_b64}" width="20" style="vertical-align: middle; margin-right: 8px;">' if away_logo_b64 else ''
        
        st.markdown(f'<div style="font-size:0.75rem; color:#f43f5e; font-weight:700; margin-bottom:5px; letter-spacing:1px; display:flex; align-items:center;">{away_img_tag}AWAY TEAM</div>', unsafe_allow_html=True)
        away_team = st.selectbox("away", team_list, index=team_list.index(away_team_val) if away_team_val in team_list else (len(team_list)-1 if team_list else 1), key="away")
        
        st.markdown('<div class="sidebar-section" style="margin-top:2rem;">ACTIVE MODELS</div>', unsafe_allow_html=True)
        st.markdown("""
            <div>
                <span class="model-tag" style="background:#1e3a8a;">XGBoost</span>
                <span class="model-tag" style="background:#064e3b; color:#34d399;">RandomForest</span>
                <span class="model-tag" style="background:#4c1d95; color:#c084fc;">GradientBoost</span>
                <span class="model-tag" style="background:#78350f; color:#fbbf24;">LogReg (Meta)</span>
            </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    # --- MAIN CONTENT ---
    # Header
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 2rem;">
        <div>
            <div class="dash-title">Match Analysis Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Needs to be called inside main() so it can access 'stats', 'home_team', 'away_team' locally
    def calculate_features(home_team, away_team, stats):
        elo_dict = stats.get('elo', {})
        history_dict = stats.get('history', {})
        mv_dict = stats.get('market_value', {})
        
        def get_form_stats(team):
            hist = history_dict.get(team, [])
            if not hist: return 0, 0, 0 
            points = sum([match['points'] for match in hist])
            gf = sum([match['gf'] for match in hist])
            ga = sum([match['ga'] for match in hist])
            return points / len(hist), gf / len(hist), ga / len(hist)

        home_elo = elo_dict.get(home_team, 1500.0)
        away_elo = elo_dict.get(away_team, 1500.0)
        h_form, h_gf, h_ga = get_form_stats(home_team)
        a_form, a_gf, a_ga = get_form_stats(away_team)
        home_mv = mv_dict.get(home_team, 0)
        away_mv = mv_dict.get(away_team, 0)

        # Simplified feature dict for prediction
        features = {
            'Home_Form_L5': h_form, 'Away_Form_L5': a_form,
            'Home_Avg_GF_L5': h_gf, 'Home_Avg_GA_L5': h_ga,
            'Away_Avg_GF_L5': a_gf, 'Away_Avg_GA_L5': a_ga,
            'Home_MV': home_mv, 'Away_MV': away_mv, 'MV_Diff': home_mv - away_mv,
            'Home_Elo': home_elo, 'Away_Elo': away_elo, 'Elo_Diff': home_elo - away_elo,
            'H2H_Home_Wins': 0, 'H2H_Away_Wins': 0, # Dummy for now
            'fpl_home': 50.0, 'fpl_away': 50.0, 'fpl_diff': 0,
            'elo_prob_home': 0.33, 'elo_prob_draw': 0.33, 'elo_prob_away': 0.33
        }
        return pd.DataFrame([features])

    input_df = calculate_features(home_team, away_team, stats)
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    # Extract probas
    prob_home = probs[0] * 100
    prob_draw = probs[1] * 100
    prob_away = probs[2] * 100

    # Map Prediction Text
    outcomes = {0: "HOME WIN", 1: "DRAW", 2: "AWAY WIN"}
    pred_text = outcomes[prediction]

    # Map Confidence
    max_prob = max(prob_home, prob_away, prob_draw)
    if max_prob > 55:
        conf_badge = '<div style="background:rgba(16,185,129,0.2); color:#10b981; border:1px solid #10b981; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">HIGH CONFIDENCE</div>'
    elif max_prob > 45:
        conf_badge = '<div style="background:rgba(245,158,11,0.2); color:#f59e0b; border:1px solid #f59e0b; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">MODERATE CONFIDENCE</div>'
    else:
        conf_badge = '<div style="background:rgba(244,63,94,0.2); color:#f43f5e; border:1px solid #f43f5e; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">RISK / UNCERTAIN</div>'


    # --- V2 Head-to-Head Card ---
    st.markdown("""
<div class="card" style="padding-bottom: 0;">
    <div class="card-title" style="justify-content:space-between; border-bottom:1px solid #2d3748; padding-bottom:15px; margin-bottom:0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#217AFA" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span>Head-to-Head Comparison</span>
        </div>
        <span class="data-source">DATA SOURCE: OPTA + TRANSFERMARKT</span>
    </div>
""", unsafe_allow_html=True)
    
    h2h_col1, h2h_col2, h2h_col3 = st.columns([1, 1, 1])
    
    elo_diff = input_df['Elo_Diff'][0]
    
    with h2h_col1:
        home_logo_path = f"app/assets/logos/{home_team.replace(' ', '_')}.png"
        home_logo_b64 = get_base64_image(home_logo_path)
        # Using logo as the main image card for now
        home_img = f'<div style="background:#0f172a; height:200px; display:flex; align-items:center; justify-content:center; border-radius:8px; margin: 30px 20px 20px 20px;"><img src="data:image/png;base64,{home_logo_b64}" style="max-height:120px; filter: drop-shadow(0 0 10px rgba(6,182,212,0.3));"></div>' if home_logo_b64 else '<div style="height:200px; background:#0f172a; margin:30px 20px 20px 20px;"></div>'
        
        st.markdown(f"""
<div style="text-align:center;">
    {home_img}
    <div class="cyan-text" style="font-size:1.6rem; margin-bottom:5px;">{home_team.upper()}</div>
    <div style="background:#1e293b; display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.6rem; color:#94a3b8; margin-bottom: 40px; border:1px solid #334155;">HOME</div>
</div>

<div style="padding: 0 20px 20px 20px;">
<div class="metric-label" style="text-align:left;">ELO RATING</div>
<div class="metric-value" style="text-align:left;">{input_df['Home_Elo'][0]:.0f}</div>
<div class="line-home" style="width:100%; margin-bottom:40px;"></div>

<div class="metric-label" style="text-align:left;">MARKET VALUE</div>
<div class="metric-value" style="font-size:1.6rem; text-align:left; margin-bottom:40px;">€{input_df['Home_MV'][0]:.0f}m</div>

<div class="metric-label" style="text-align:left;">FORM (LAST 5)</div>
<div class="metric-value" style="font-size:1.6rem; text-align:left; color:#94a3b8;">{input_df['Home_Form_L5'][0]:.2f}</div>
</div>
""", unsafe_allow_html=True)
        
    with h2h_col2:
        st.markdown(f"""
<div style="height: 100%; display:flex; flex-direction:column; justify-content:flex-start; align-items:center; margin-top:140px;">
    <div style="width:80px; height:80px; border-radius:50%; border:2px solid #334155; background:#0f172a; display:flex; align-items:center; justify-content:center; font-family:'Inter', sans-serif; font-style:italic; font-size:1.8rem; color:#94a3b8; margin-bottom:30px;">VS</div>
    <div style="background:#161B22; border-radius:8px; padding:15px 30px; text-align:center; border:1px solid #2d3748; margin-bottom:15px; width:180px;">
        <div style="font-size:0.65rem; color:#64748b; margin-bottom:5px; text-transform:uppercase;">ELO DIFFERENCE</div>
        <div style="font-family:'Inter', sans-serif; font-size:1.8rem; font-weight:800; color:{'#06b6d4' if elo_diff > 0 else '#f43f5e'};">{'+' if elo_diff > 0 else ''}{elo_diff:.0f}</div>
    </div>
    <div style="font-size:0.6rem; color:#64748b; text-align:center; text-transform:uppercase; line-height:1.4;">Historical Rivalry<br>Matchday 26</div>
</div>
""", unsafe_allow_html=True)
        
    with h2h_col3:
        away_logo_path = f"app/assets/logos/{away_team.replace(' ', '_')}.png"
        away_logo_b64 = get_base64_image(away_logo_path)
        away_img = f'<div style="background:#0f172a; height:200px; display:flex; align-items:center; justify-content:center; border-radius:8px; margin: 30px 20px 20px 20px;"><img src="data:image/png;base64,{away_logo_b64}" style="max-height:120px; filter: drop-shadow(0 0 10px rgba(244,63,94,0.3));"></div>' if away_logo_b64 else '<div style="height:200px; background:#0f172a; margin:30px 20px 20px 20px;"></div>'
        
        st.markdown(f"""
<div style="text-align:center;">
    {away_img}
    <div class="red-text" style="font-size:1.6rem; margin-bottom:5px;">{away_team.upper()}</div>
    <div style="background:#1e293b; display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.6rem; color:#94a3b8; margin-bottom: 40px; border:1px solid #334155;">AWAY</div>
</div>

<div style="padding: 0 20px 20px 20px;">
<div class="metric-label" style="text-align:right;">ELO RATING</div>
<div class="metric-value" style="text-align:right;">{input_df['Away_Elo'][0]:.0f}</div>
<div class="line-away" style="width:100%; margin-bottom:40px;"></div>

<div class="metric-label" style="text-align:right;">MARKET VALUE</div>
<div class="metric-value" style="font-size:1.6rem; text-align:right; margin-bottom:40px;">€{input_df['Away_MV'][0]:.0f}m</div>

<div class="metric-label" style="text-align:right;">FORM (LAST 5)</div>
<div class="metric-value" style="font-size:1.6rem; text-align:right; color:#94a3b8;">{input_df['Away_Form_L5'][0]:.2f}</div>
</div>
""", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- NEW: AI Prediction Engine & Comparative Feature Analysis ---
    comp_col1, comp_col2 = st.columns([1.1, 0.9], gap="large")

    with comp_col1:
        # Determine winning prediction styles
        pred_home_win = max_prob == prob_home
        pred_draw = max_prob == prob_draw
        pred_away_win = max_prob == prob_away

        st.markdown(f"""
<div class="card" style="height: 100%; padding:25px;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:40px;">
<div style="display:flex; align-items:center; gap:10px;">
<div style="width:4px; height:24px; background:#10b981; border-radius:4px; box-shadow:0 0 10px rgba(16,185,129,0.5);"></div>
<span style="font-size:1.6rem; font-weight:800; color:#f8fafc; letter-spacing:0.5px; font-family:'Inter', sans-serif; text-transform:uppercase;">AI PREDICTION ENGINE</span>
</div>
</div>
<div style="display:flex; gap:10px; margin-top:-25px; margin-bottom:30px; align-items:center;">
{conf_badge}
<span style="font-size:0.6rem; color:#64748b; font-family:'Inter', sans-serif;">MODEL: ENSEMBLE V5.2</span>
</div>

<!-- Home Win Probability -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="display:flex; align-items:center; gap:15px;">
<div style="width:40px; height:40px; border-radius:8px; border:1px solid #1e293b; background:#0f172a; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden;">
<span style="font-size:1.2rem; filter: sepia(100%) hue-rotate(180deg) saturate(300%) opacity(0.8);">🏃</span>
</div>
<div>
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">OUTCOME</div>
<div style="font-size:1.2rem; font-weight:700; color:{'#06b6d4' if pred_home_win else '#e2e8f0'}; font-family:'Inter', sans-serif; text-transform:uppercase;">{home_team} WIN</div>
</div>
</div>
<div style="font-size:2rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_home:.1f}%</div>
</div>
<div style="width:100%; height:12px; background:#1e293b; border-radius:6px; border:1px solid #334155; overflow:hidden;">
<div style="width:{prob_home}%; height:100%; background:{'linear-gradient(90deg, #06b6d4, #10b981)' if pred_home_win else '#475569'}; border-radius:6px; box-shadow:{'0 0 15px rgba(16,185,129,0.5)' if pred_home_win else 'none'};"></div>
</div>
</div>

<!-- Match Draw Probability -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="display:flex; align-items:center; gap:15px;">
<div style="width:40px; height:40px; border-radius:8px; border:1px solid #1e293b; background:#0f172a; display:flex; align-items:center; justify-content:center;">
<span style="font-size:1.4rem; color:#64748b;">=</span>
</div>
<div>
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">STABILITY</div>
<div style="font-size:1.2rem; font-weight:700; color:{'#06b6d4' if pred_draw else '#e2e8f0'}; font-family:'Inter', sans-serif;">MATCH DRAW</div>
</div>
</div>
<div style="font-size:2rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_draw:.1f}%</div>
</div>
<div style="width:100%; height:12px; background:#1e293b; border-radius:6px; border:1px solid #334155; overflow:hidden;">
<div style="width:{prob_draw}%; height:100%; background:{'linear-gradient(90deg, #06b6d4, #10b981)' if pred_draw else '#475569'}; border-radius:6px; box-shadow:{'0 0 15px rgba(16,185,129,0.5)' if pred_draw else 'none'};"></div>
</div>
</div>

<!-- Away Win Probability -->
<div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="display:flex; align-items:center; gap:15px;">
<div style="width:40px; height:40px; border-radius:8px; border:1px solid #1e293b; background:#0f172a; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden;">
<span style="font-size:1.2rem; filter: sepia(100%) hue-rotate(0deg) saturate(300%) opacity(0.8);">🏙️</span>
</div>
<div>
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">RISK</div>
<div style="font-size:1.2rem; font-weight:700; color:{'#06b6d4' if pred_away_win else '#f43f5e'}; font-family:'Inter', sans-serif; text-transform:uppercase;">{away_team} WIN</div>
</div>
</div>
<div style="font-size:2rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_away:.1f}%</div>
</div>
<div style="width:100%; height:12px; background:#1e293b; border-radius:6px; border:1px solid #334155; overflow:hidden;">
<div style="width:{prob_away}%; height:100%; background:{'linear-gradient(90deg, #06b6d4, #10b981)' if pred_away_win else '#f43f5e'}; border-radius:6px; box-shadow:{'0 0 15px rgba(244,63,94,0.5)' if pred_away_win else 'none'};"></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    with comp_col2:
        # Mocking data for comparative bars
        home_elo = input_df['Home_Elo'][0]
        away_elo = input_df['Away_Elo'][0]
        max_elo = max(home_elo, away_elo) * 1.1
        
        home_mv_display = f"€{input_df['Home_MV'][0]/1000:.2f}BN" if input_df['Home_MV'][0] >= 1000 else f"€{input_df['Home_MV'][0]:.0f}M"
        away_mv_display = f"€{input_df['Away_MV'][0]/1000:.2f}BN" if input_df['Away_MV'][0] >= 1000 else f"€{input_df['Away_MV'][0]:.0f}M"
        max_mv = max(input_df['Home_MV'][0], input_df['Away_MV'][0]) * 1.1

        home_form = input_df['Home_Form_L5'][0]
        away_form = input_df['Away_Form_L5'][0]
        max_form = max(home_form, away_form, 1) * 1.2
        
        home_goals = input_df['Home_Avg_GF_L5'][0]
        away_goals = input_df['Away_Avg_GF_L5'][0]
        max_goals = max(home_goals, away_goals, 0.1) * 1.2
        
        # Dynamic Delta Analysis Logic
        h_adv = []
        a_adv = []
        
        if home_goals > away_goals * 1.1:
            h_adv.append(f"offensive output (+{((home_goals-away_goals)/max(0.1, away_goals))*100:.0f}% Goals)")
        elif away_goals > home_goals * 1.1:
            a_adv.append(f"offensive output (+{((away_goals-home_goals)/max(0.1, home_goals))*100:.0f}% Goals)")
            
        home_ga = input_df['Home_Avg_GA_L5'][0]
        away_ga = input_df['Away_Avg_GA_L5'][0]
        if home_ga < away_ga * 0.9:
            h_adv.append("defensive solidity")
        elif away_ga < home_ga * 0.9:
            a_adv.append("defensive solidity")
            
        mv_diff = input_df['MV_Diff'][0]
        if mv_diff > 50_000_000:
            h_adv.append("squad valuation")
        elif mv_diff < -50_000_000:
            a_adv.append("squad valuation")
            
        if home_form > away_form + 0.3:
            h_adv.append("recent form")
        elif away_form > home_form + 0.3:
            a_adv.append("recent form")
            
        if not h_adv and not a_adv:
            delta_text = "Both teams share very similar statistical profiles across key metrics, making this a tightly contested matchup."
        else:
            parts = []
            if h_adv:
                parts.append(f"Home team shows strength in {', '.join(h_adv)}")
            if a_adv:
                parts.append(f"Away team holds an edge in {', '.join(a_adv)}")
            delta_text = " while ".join(parts).capitalize() + "."

        st.markdown(f"""
<div class="card" style="height: 100%; padding:25px;">
<div style="font-size:1.1rem; color:#94a3b8; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing:1px; margin-bottom:30px; line-height:1.4;">
COMPARATIVE FEATURE<br>ANALYSIS
</div>

<!-- ELO Rating -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-weight:700; font-family:'Inter', sans-serif; margin-bottom:6px;">
<div style="color:#06b6d4;">{home_elo:,.0f}</div>
<div style="color:#475569; letter-spacing:1px;">ELO RATING</div>
<div style="color:#f43f5e;">{away_elo:,.0f}</div>
</div>
<div style="display:flex; gap:10px; height:8px;">
<div style="flex:1; background:#1e293b; border-radius:4px; display:flex; justify-content:flex-end; overflow:hidden;">
<div style="width:{(home_elo/max_elo)*100}%; background:#06b6d4; border-radius:4px;"></div>
</div>
<div style="flex:1; background:#1e293b; border-radius:4px; overflow:hidden;">
<div style="width:{(away_elo/max_elo)*100}%; background:#f43f5e; border-radius:4px;"></div>
</div>
</div>
</div>

<!-- Market Value -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-weight:700; font-family:'Inter', sans-serif; margin-bottom:6px;">
<div style="color:#06b6d4;">{home_mv_display}</div>
<div style="color:#475569; letter-spacing:1px;">MARKET VALUE</div>
<div style="color:#f43f5e;">{away_mv_display}</div>
</div>
<div style="display:flex; gap:10px; height:8px;">
<div style="flex:1; background:#1e293b; border-radius:4px; display:flex; justify-content:flex-end; overflow:hidden;">
<div style="width:{(input_df['Home_MV'][0]/max_mv)*100}%; background:#06b6d4; border-radius:4px; opacity:0.7;"></div>
</div>
<div style="flex:1; background:#1e293b; border-radius:4px; overflow:hidden;">
<div style="width:{(input_df['Away_MV'][0]/max_mv)*100}%; background:#f43f5e; border-radius:4px;"></div>
</div>
</div>
</div>

<!-- Form -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-weight:700; font-family:'Inter', sans-serif; margin-bottom:6px;">
<div style="color:#06b6d4;">{home_form:.2f}</div>
<div style="color:#475569; letter-spacing:1px;">RECENT FORM</div>
<div style="color:#f43f5e;">{away_form:.2f}</div>
</div>
<div style="display:flex; gap:10px; height:8px;">
<div style="flex:1; background:#1e293b; border-radius:4px; display:flex; justify-content:flex-end; overflow:hidden;">
<div style="width:{(home_form/max_form)*100}%; background:#06b6d4; border-radius:4px;"></div>
</div>
<div style="flex:1; background:#1e293b; border-radius:4px; overflow:hidden;">
<div style="width:{(away_form/max_form)*100}%; background:#f43f5e; border-radius:4px; opacity:0.6;"></div>
</div>
</div>
</div>

<!-- Goals -->
<div style="margin-bottom:40px;">
<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-weight:700; font-family:'Inter', sans-serif; margin-bottom:6px;">
<div style="color:#06b6d4;">{home_goals:.1f}</div>
<div style="color:#475569; letter-spacing:1px;">AVG GOALS</div>
<div style="color:#f43f5e;">{away_goals:.1f}</div>
</div>
<div style="display:flex; gap:10px; height:8px;">
<div style="flex:1; background:#1e293b; border-radius:4px; display:flex; justify-content:flex-end; overflow:hidden;">
<div style="width:{(home_goals/max_goals)*100}%; background:#06b6d4; border-radius:4px;"></div>
</div>
<div style="flex:1; background:#1e293b; border-radius:4px; overflow:hidden;">
<div style="width:{(away_goals/max_goals)*100}%; background:#f43f5e; border-radius:4px; opacity:0.7;"></div>
</div>
</div>
</div>

<!-- Delta Analysis -->
<div style="border-top:1px solid #1e293b; padding-top:20px;">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
<span style="color:#06b6d4;">📊</span>
<span style="font-size:0.75rem; font-weight:700; color:#06b6d4; letter-spacing:1px;">DELTA ANALYSIS</span>
</div>
<div style="font-size:0.8rem; color:#94a3b8; line-height:1.5; font-family:'Inter', sans-serif;">
{delta_text}
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- Confidence Calibration (V2 Bars) ---
    # Mocking Fan Data
    fan_home = max(0, min(100, prob_home + (elo_diff * 0.05)))
    fan_away = max(0, min(100, prob_away - (elo_diff * 0.05)))
    fan_draw = 100 - fan_home - fan_away
    
    diff_home = prob_home - fan_home
    leading_text = f"AI LEADING BY {diff_home:+.2f}% OVER CROWD" if abs(diff_home) > 0.5 else "AI ALIGNED WITH CROWD CONSENSUS"
    lead_color = "#10b981" if diff_home > 0 else "#f59e0b"

    st.markdown("""
<div class="card-title" style="justify-content:space-between; margin-bottom:30px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.2rem;">⚖️</span>
        <span style="font-size:1.4rem; color:#f8fafc;">Confidence Calibration</span>
    </div>
    <div style="background:rgba(16,185,129,0.15); color:""" + lead_color + """; border:1px solid rgba(16,185,129,0.3); padding:4px 12px; border-radius:4px; font-size:0.65rem; font-weight:700; font-family:'Inter', sans-serif;">
        ⚡ """ + leading_text + """
    </div>
</div>
""", unsafe_allow_html=True)

    # Home Win
    st.markdown(f"""
<div style="margin-bottom:25px;">
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase;">{home_team} Win Probabilities</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_home:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">FAN CONFIDENCE</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; font-family:'Inter', sans-serif;">{fan_home:.1f}%</div>
            </div>
        </div>
    </div>
    <div style="width:100%; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; display:flex;">
        <div style="width:{prob_home}%; height:100%; background:#10b981; border-radius:4px 0 0 4px;"></div>
        <div style="width:{max(0, fan_home - prob_home)}%; height:100%; background:#2563eb; opacity:0.6;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Draw
    st.markdown(f"""
<div style="margin-bottom:25px;">
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px;">DRAW PROBABILITIES</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_draw:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">FAN CONFIDENCE</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; font-family:'Inter', sans-serif;">{fan_draw:.1f}%</div>
            </div>
        </div>
    </div>
    <div style="width:100%; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; display:flex;">
        <div style="width:{prob_draw}%; height:100%; background:#94a3b8; border-radius:4px 0 0 4px;"></div>
        <div style="width:{max(0, fan_draw - prob_draw)}%; height:100%; background:#2563eb; opacity:0.6;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Away Win
    st.markdown(f"""
<div style="margin-bottom:40px;">
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;">
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase;">{away_team} Win Probabilities</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_away:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">FAN CONFIDENCE</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; font-family:'Inter', sans-serif;">{fan_away:.1f}%</div>
            </div>
        </div>
    </div>
    <div style="width:100%; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; display:flex;">
        <div style="width:{prob_away}%; height:100%; background:#f43f5e; border-radius:4px 0 0 4px;"></div>
        <div style="width:{max(0, fan_away - prob_away)}%; height:100%; background:#2563eb; opacity:0.6;"></div>
    </div>
</div>

<div style="display:flex; align-items:center; gap:10px; padding-top:15px; border-top:1px solid #1f2937;">
    <div style="color:#64748b;">ⓘ</div>
    <div style="font-size:0.75rem; color:#64748b; font-style:italic;">Human Fan Confidence is derived from current FPL ownership data and live social sentiment analysis (Mocked).</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- AI vs Human Performance Benchmark ---
    st.markdown("""
<div class="card">
<div class="card-title" style="margin-bottom:30px;">
<div style="display:flex; align-items:center; gap:10px;">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="w-6 h-6"><path d="M4 14V20M8 8V20M12 12V20M16 16V20M20 4V20" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
<span style="font-size:1.4rem; color:#f8fafc; font-weight:800; font-family:'Orbitron', sans-serif;">AI vs Human Performance Benchmark</span>
</div>
</div>

<div style="display:flex; gap:20px; margin-bottom:20px;">
<!-- Card 1 -->
<div style="flex:1; border:1px solid #064e3b; border-radius:8px; padding:20px; background:rgba(6,78,59,0.1); position:relative;">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#10b981; font-weight:700; margin-bottom:5px; letter-spacing:1px;">OUR MODEL (V5)</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">55.26%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#10b981; display:flex; align-items:center; gap:5px;">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
Highest Accuracy
</div>
</div>

<!-- Card 2 -->
<div style="flex:1; border:1px solid #1e3a8a; border-radius:8px; padding:20px; background:rgba(30,58,138,0.1);">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#60a5fa; font-weight:700; margin-bottom:5px; letter-spacing:1px;">HUMAN FANS (FPL)</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">54.74%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#94a3b8;">
Baseline Target
</div>
</div>

<!-- Card 3 -->
<div style="flex:1; border:1px solid #4c1d95; border-radius:8px; padding:20px; background:rgba(76,29,149,0.1);">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#c084fc; font-weight:700; margin-bottom:5px; letter-spacing:1px;">BBC EXPERTS</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">47.11%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect><path d="M9 22v-4h6v4"></path><path d="M8 6h.01"></path><path d="M16 6h.01"></path><path d="M12 6h.01"></path><path d="M12 10h.01"></path><path d="M12 14h.01"></path><path d="M16 10h.01"></path><path d="M16 14h.01"></path><path d="M8 10h.01"></path><path d="M8 14h.01"></path></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#94a3b8;">
Standard Metric
</div>
</div>
</div>

<!-- Banner -->
<div style="background:rgba(6,78,59,0.2); border:1px solid #064e3b; border-radius:8px; padding:15px; display:flex; align-items:center; gap:10px;">
<span style="color:#fbbf24; font-size:1.2rem;">🏆</span>
<span style="color:#10b981; font-weight:600; font-size:0.95rem; font-family:'Inter', sans-serif;">Result: AI beats all baselines! More accurate than Fan Confidence (+0.52%) and BBC Experts (+8.15%).</span>
</div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
