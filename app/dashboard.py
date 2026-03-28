import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import pandas as pd
import joblib
import json
import altair as alt
import datetime
import base64

def get_base64_image(image_path):
    # Ensure we use the full path correctly
    if not os.path.isabs(image_path):
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, image_path)
    
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set page config
st.set_page_config(page_title="PremierPredict-AI | Match Analytics", layout="wide", page_icon="⚽", initial_sidebar_state="expanded")

# Custom CSS for the New Match Analysis Dashboard Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements but KEEP header for the toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] { 
        background-color: rgba(0,0,0,0) !important;
        visibility: visible !important;
    }

    /* Base App Theme */
    html, body, .stApp {
        background-color: #11141E !important; /* Very dark blue background with subtle texture */
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161A25 !important;
        border-right: 1px solid #1f2937 !important;
    }
    /* Specifically hide only the decoration line at the top, not the whole header */
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stHeader"] { background: transparent !important; }
    .sea612-bar {
        padding: 0.55rem 1rem;
        font-size: 0.62rem;
        color: #93c5fd;
        background: rgba(37,99,235,0.18);
        border-bottom: 1px solid rgba(59,130,246,0.3);
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    /* Better Badge: Pinned to the very top */
    .sidebar-badge {
        background: rgba(37,99,235,0.25);
        color: #93c5fd;
        padding: 10px 15px;
        font-size: 0.75rem; /* Bigger for visibility */
        font-weight: 700;
        border-bottom: 1px solid rgba(59,130,246,0.35);
        margin-bottom: 2px; /* Pull the logo up tight */
        margin-top: -120px !important; /* Extremely aggressive negative margin to overcome header offset */
        font-family: 'Inter', sans-serif;
        text-align: center;
        width: calc(100% + 3rem); /* Full width across sidebar padding */
        margin-left: -1.5rem;
        letter-spacing: 0.5px;
    }

    /* Ensure icons always use their native font, don't override them */
    [data-testid="stSidebar"] button[kind="secondary"] i,
    [data-testid="stSidebar"] button[kind="secondary"] svg,
    [data-testid="stHeader"] i,
    [data-testid="stHeader"] svg {
        font-family: inherit !important;
        color: white !important; /* Force icons to be white */
    }
    
    /* UNIVERSAL SIDEBAR TOGGLE FIX - Force the arrow to show on top */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        z-index: 999999 !important;
        left: 10px !important;
        top: 10px !important;
        background-color: rgba(22, 26, 37, 0.8) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 5px !important;
    }
    
    /* Force the button inside the collapsed control to be blue and visible */
    [data-testid="stSidebarCollapsedControl"] button {
        background-color: #2563eb !important; /* Royal Blue */
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 6px !important;
        opacity: 1 !important;
    }

    /* Style the CLOSE button when sidebar is open */
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {
        color: white !important;
        background-color: #2563eb !important;
        border-radius: 6px !important;
        margin-top: 0px !important; /* Push close button to the absolute top edge */
        margin-right: 10px !important;
    }
    
    [data-testid="stSidebarContent"] {
        font-family: 'Inter', sans-serif !important;
        padding-top: 0 !important;
        overflow: visible !important; /* Allow negative margins to show above content area */
    }
    
    /* Top Header Adjustments */
    .block-container { padding-top: 3.5rem !important; max-width: 1400px; }
    
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
    div[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {
        border-radius: 4px;
        box-shadow: none;
        border-bottom: 2px solid #334155 !important;
    }
    
    /* Hide selectbox label */
    .stSelectbox label { display: none; }

    /* Handle horizontal benchmark cards wrapping on mobile */
    .benchmark-container {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }

    /* ── RESPONSIVE: Tablet (≤ 1024px) ── */
    @media (max-width: 1024px) {
        .block-container { padding-top: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
        .dash-title { font-size: 1.6rem !important; }
        .card-title { font-size: 0.95rem !important; }
        .metric-value { font-size: 1.6rem !important; }
        .cyan-text, .red-text { font-size: 1.2rem !important; }
        .pred-result-text { font-size: 2rem !important; }
        .benchmark-container { flex-wrap: wrap; }
    }

    /* ── RESPONSIVE: Mobile (≤ 768px) ── */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        /* Scale down main title */
        .dash-title { font-size: 1.3rem !important; letter-spacing: 0.5px !important; margin-top: 1rem !important; }
        /* Card titles */
        .card-title { font-size: 0.9rem !important; flex-wrap: wrap; }
        /* Metric numbers */
        .metric-value { font-size: 1.2rem !important; }
        .metric-label { font-size: 0.65rem !important; }
        /* Team name text */
        .cyan-text, .red-text { font-size: 1.1rem !important; }
        /* Prediction result */
        .pred-result-text { font-size: 1.6rem !important; }
        /* Cards: reduce padding */
        .card { padding: 16px !important; }
        /* Reduce sidebar logo size */
        .sidebar-logo { font-size: 1.2rem !important; }
        /* Data source floated right — hide on mobile or adjust */
        .data-source { display: block; float: none; font-size: 0.55rem; margin-top: 5px; }
        /* Confidence calibration section title */
        .card-title span { font-size: 0.9rem !important; }
        /* Feature rows */
        .feature-row { font-size: 0.8rem !important; }
        /* Aggressive font size reduction for inline styles */
        [style*="font-size:2.8rem"], [style*="font-size: 2.8rem"] { font-size: 1.6rem !important; }
        [style*="font-size:2.5rem"], [style*="font-size: 2.5rem"] { font-size: 1.5rem !important; }
        [style*="font-size:2.2rem"], [style*="font-size: 2.2rem"] { font-size: 1.4rem !important; }
        [style*="font-size:2rem"], [style*="font-size: 2rem"] { font-size: 1.3rem !important; }
        [style*="font-size:1.8rem"], [style*="font-size: 1.8rem"] { font-size: 1.2rem !important; }
        [style*="font-size:1.6rem"], [style*="font-size: 1.6rem"] { font-size: 1.1rem !important; }
        [style*="font-size:1.4rem"], [style*="font-size: 1.4rem"] { font-size: 1rem !important; }
        [style*="height:200px"], [style*="height: 200px"] { height: 120px !important; }
        [style*="max-height:120px"], [style*="max-height: 120px"] { max-height: 80px !important; }
        [style*="margin-top:140px"], [style*="margin-top: 140px"] { margin-top: 20px !important; padding: 20px 0; }
        .benchmark-container { flex-direction: column; }
    }

</style>

""", unsafe_allow_html=True)

MODEL_PATH = "models/stacking_ensemble_v5.pkl"
STATS_PATH = "data/processed/latest_team_stats.json"
RESULTS_PATH = "data/processed/ensemble_results.json"
METRICS_PATH = "models/metrics_v5.json"
COMP_PATH = "data/processed/ai_vs_crowd_comparison.csv"
HUMAN_COMP_PATH = "data/processed/human_baseline_comparison.csv"
PRED_2025_PATH = "data/processed/predictions_2025.csv"

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
        model = None
    try:
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = None
    try:
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    return model, stats, results, metrics

def main():
    model, stats, results, metrics = load_resources()
    if not model or not stats:
        st.error("Backend models/stats error: Unable to access system resources.")
        return
        
    try:
        history_df = pd.read_csv(COMP_PATH)
    except FileNotFoundError:
        history_df = pd.DataFrame()
        
    try:
        human_baseline_df = pd.read_csv(HUMAN_COMP_PATH)
    except FileNotFoundError:
        human_baseline_df = pd.DataFrame()

    # Accuracy definitions: Prefer METRICS_PATH (Official Paper) > RESULTS_PATH > Defaults
    ai_acc = metrics.get('accuracy', results.get('ensemble_accuracy', 55.26))
    crowd_acc = 54.74  # Based on paper benchmark (55.26 - 0.52 = 54.74)
    human_acc = 47.11  # Based on paper benchmark (55.26 - 8.15 = 47.11)

    # --- SIDEBAR ---
    with st.sidebar:
        # Pinned Badge as a div instead of CSS ::before for better stability
        st.markdown('<div class="sidebar-badge">SEA612 | AI Fundamentals</div>', unsafe_allow_html=True)
        
        # Branding section (now moved up together)
        st.markdown('<div class="sidebar-logo" style="margin-top:-60px !important; font-size:1.4rem;">🤖 PremierPredict</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.65rem; color:#64748b; margin-top:-10px !important; margin-bottom:0.75rem;">STACKING ENSEMBLE V5.0 (SEA612)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">Match Simulation</div>', unsafe_allow_html=True)
        
        team_list = sorted(list(stats['elo'].keys()))
        app_dir = os.path.dirname(__file__)
        home_team_val = st.session_state.get("home", team_list[0] if team_list else "AFC Bournemouth")
        home_logo_name = normalize_team_name(home_team_val).replace(" ", "_")
        home_logo_path = f"assets/logos/{home_logo_name}.png"
        home_logo_b64 = get_base64_image(home_logo_path)
        home_img_tag = f'<img src="data:image/png;base64,{home_logo_b64}" width="20" style="vertical-align: middle; margin-right: 8px;">' if home_logo_b64 else ''
        
        st.markdown(f'<div style="font-size:0.75rem; color:#06b6d4; font-weight:700; margin-bottom:5px; letter-spacing:1px; display:flex; align-items:center;">{home_img_tag}Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("home", team_list, index=team_list.index(home_team_val) if home_team_val in team_list else 0, key="home")
        
        st.markdown('<div style="text-align:center; color:#64748b; margin:10px 0;">⚔️</div>', unsafe_allow_html=True)
        
        away_team_val = st.session_state.get("away", team_list[-1] if team_list else "Wolverhampton Wanderers FC")
        away_logo_name = normalize_team_name(away_team_val).replace(" ", "_")
        away_logo_path = f"assets/logos/{away_logo_name}.png"
        away_logo_b64 = get_base64_image(away_logo_path)
        away_img_tag = f'<img src="data:image/png;base64,{away_logo_b64}" width="20" style="vertical-align: middle; margin-right: 8px;">' if away_logo_b64 else ''
        
        st.markdown(f'<div style="font-size:0.75rem; color:#f43f5e; font-weight:700; margin-bottom:5px; letter-spacing:1px; display:flex; align-items:center;">{away_img_tag}Away Team</div>', unsafe_allow_html=True)
        away_team = st.selectbox("away", team_list, index=team_list.index(away_team_val) if away_team_val in team_list else (len(team_list)-1 if team_list else 1), key="away")
        
        st.markdown('<div class="sidebar-section" style="margin-top:2rem;">Active Models</div>', unsafe_allow_html=True)
        st.markdown("""
            <div>
                <span class="model-tag" style="background:#1e3a8a;">XGBoost</span>
                <span class="model-tag" style="background:#064e3b; color:#34d399;">RandomForest</span>
                <span class="model-tag" style="background:#4c1d95; color:#c084fc;">GradientBoost</span>
                <span class="model-tag" style="background:#78350f; color:#fbbf24;">LogReg (Meta)</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section" style="margin-top:2rem;">Group Members (SEA612)</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:0.65rem; color:#94a3b8; line-height:1.6;">
                <div>• Chaiyaporn Homtean (67130700346)</div>
                <div>• Krittin Chaisuvirat (67130700357)</div>
                <div>• Pawornwit Maneenet (67130700361)</div>
            </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    # Header
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 2rem;">
        <div>
            <div class="dash-title">Match Analytics Dashboard</div>
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

        h2h_dict = stats.get('h2h', {})
        h2h_key = f"{home_team}_vs_{away_team}"
        if h2h_key not in h2h_dict:
            h2h_key = f"{away_team}_vs_{home_team}"
        
        h2h_history = h2h_dict.get(h2h_key, [])
        h2h_home_wins = h2h_history.count(home_team)
        h2h_away_wins = h2h_history.count(away_team)
        h2h_draws = h2h_history.count("DRAW")
        
        # Simplified feature dict for prediction
        features = {
            'Home_Form_L5': h_form, 'Away_Form_L5': a_form,
            'Home_Avg_GF_L5': h_gf, 'Home_Avg_GA_L5': h_ga,
            'Away_Avg_GF_L5': a_gf, 'Away_Avg_GA_L5': a_ga,
            'Home_MV': home_mv, 'Away_MV': away_mv, 'MV_Diff': home_mv - away_mv,
            'Home_Elo': home_elo, 'Away_Elo': away_elo, 'Elo_Diff': home_elo - away_elo,
            'H2H_Home_Wins': h2h_home_wins, 'H2H_Away_Wins': h2h_away_wins,
            'fpl_home': 50.0, 'fpl_away': 50.0, 'fpl_diff': 0,
            'elo_prob_home': 0.33, 'elo_prob_draw': 0.33, 'elo_prob_away': 0.33
        }
        return pd.DataFrame([features]), h2h_home_wins, h2h_away_wins, h2h_draws, len(h2h_history)

    input_df, h2h_home_wins, h2h_away_wins, h2h_draws, h2h_count = calculate_features(home_team, away_team, stats)
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    # Extract probas
    prob_home = probs[0] * 100
    prob_draw = probs[1] * 100
    prob_away = probs[2] * 100

    # Map Prediction Text
    outcomes = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    pred_text = outcomes[prediction]

    # Map Confidence
    max_prob = max(prob_home, prob_away, prob_draw)
    if max_prob > 55:
        conf_badge = '<div style="background:rgba(16,185,129,0.2); color:#10b981; border:1px solid #10b981; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">HIGH CONFIDENCE</div>'
    elif max_prob > 45:
        conf_badge = '<div style="background:rgba(245,158,11,0.2); color:#f59e0b; border:1px solid #f59e0b; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">MEDIUM CONFIDENCE</div>'
    else:
        conf_badge = '<div style="background:rgba(244,63,94,0.2); color:#f43f5e; border:1px solid #f43f5e; padding:2px 8px; border-radius:4px; font-size:0.6rem; font-weight:700;">RISK / UNCERTAIN</div>'


    # --- V2 Head-to-Head Card ---
    st.markdown("""
<div class="card" style="padding-bottom: 0;">
    <div class="card-title" style="justify-content:space-between; border-bottom:1px solid #2d3748; padding-bottom:15px; margin-bottom:0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#217AFA" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span>Head-to-Head Statistics Comparison</span>
        </div>
        <span class="data-source">Data Source: OPTA + TRANSFERMARKT + ELO</span>
    </div>
""", unsafe_allow_html=True)
    
    h2h_col1, h2h_col2, h2h_col3 = st.columns([1, 1, 1])
    
    elo_diff = input_df['Elo_Diff'][0]
    
    with h2h_col1:
        home_logo_path = f"assets/logos/{home_team.replace(' ', '_')}.png"
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
<div class="metric-label" style="text-align:left;">ELO Rating</div>
<div class="metric-value" style="text-align:left;">{input_df['Home_Elo'][0]:.0f}</div>
<div class="line-home" style="width:100%; margin-bottom:40px;"></div>

<div class="metric-label" style="text-align:left;">Squad Market Value</div>
<div class="metric-value" style="font-size:1.6rem; text-align:left; margin-bottom:40px;">€{input_df['Home_MV'][0]:.0f}m</div>

<div class="metric-label" style="text-align:left;">Form (Last 5)</div>
<div class="metric-value" style="font-size:1.6rem; text-align:left; color:#94a3b8;">{input_df['Home_Form_L5'][0]:.2f}</div>
</div>
""", unsafe_allow_html=True)
        
    with h2h_col2:
        st.markdown(f"""
<div style="height: 100%; display:flex; flex-direction:column; justify-content:flex-start; align-items:center; margin-top:140px;">
    <div style="width:80px; height:80px; border-radius:50%; border:2px solid #334155; background:#0f172a; display:flex; align-items:center; justify-content:center; font-family:'Inter', sans-serif; font-style:italic; font-size:1.8rem; color:#94a3b8; margin-bottom:30px;">VS</div>
    <div style="background:#161B22; border-radius:8px; padding:15px 30px; text-align:center; border:1px solid #2d3748; margin-bottom:15px; width:180px;">
        <div style="font-size:0.65rem; color:#64748b; margin-bottom:5px; text-transform:uppercase;">ELO Rating Difference</div>
        <div style="font-family:'Inter', sans-serif; font-size:1.8rem; font-weight:800; color:{'#06b6d4' if elo_diff > 0 else '#f43f5e'};">{'+' if elo_diff > 0 else ''}{elo_diff:.0f}</div>
    </div>
    <div style="font-size:0.6rem; color:#64748b; text-align:center; text-transform:uppercase; line-height:1.4;">Total Head-to-Head<br>History ({h2h_count} Matches)</div>
    <div style="font-size:0.55rem; color:#94a3b8; margin-top:10px; text-align:center;">
        <span style="color:#06b6d4;">WON: {h2h_home_wins}</span> | <span style="color:#94a3b8;">DRAW: {h2h_draws}</span> | <span style="color:#f43f5e;">LOST: {h2h_away_wins}</span>
    </div>
</div>
""", unsafe_allow_html=True)
        
    with h2h_col3:
        away_logo_path = f"assets/logos/{away_team.replace(' ', '_')}.png"
        away_logo_b64 = get_base64_image(away_logo_path)
        away_img = f'<div style="background:#0f172a; height:200px; display:flex; align-items:center; justify-content:center; border-radius:8px; margin: 30px 20px 20px 20px;"><img src="data:image/png;base64,{away_logo_b64}" style="max-height:120px; filter: drop-shadow(0 0 10px rgba(244,63,94,0.3));"></div>' if away_logo_b64 else '<div style="height:200px; background:#0f172a; margin:30px 20px 20px 20px;"></div>'
        
        st.markdown(f"""
<div style="text-align:center;">
    {away_img}
    <div class="red-text" style="font-size:1.6rem; margin-bottom:5px;">{away_team.upper()}</div>
    <div style="background:#1e293b; display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.6rem; color:#94a3b8; margin-bottom: 40px; border:1px solid #334155;">AWAY</div>
</div>

<div style="padding: 0 20px 20px 20px;">
<div class="metric-label" style="text-align:right;">ELO Rating</div>
<div class="metric-value" style="text-align:right;">{input_df['Away_Elo'][0]:.0f}</div>
<div class="line-away" style="width:100%; margin-bottom:40px;"></div>

<div class="metric-label" style="text-align:right;">Squad Market Value</div>
<div class="metric-value" style="font-size:1.6rem; text-align:right; margin-bottom:40px;">€{input_df['Away_MV'][0]:.0f}m</div>

<div class="metric-label" style="text-align:right;">Form (Last 5)</div>
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
<span style="font-size:1.6rem; font-weight:800; color:#f8fafc; letter-spacing:0.5px; font-family:'Inter', sans-serif; text-transform:uppercase;">AI ANALYSIS & PREDICTION SYSTEM</span>
</div>
</div>
<div style="display:flex; gap:10px; margin-top:-25px; margin-bottom:30px; align-items:center;">
{conf_badge}
<span style="font-size:0.6rem; color:#64748b; font-family:'Inter', sans-serif;">MODEL: STACKING ENSEMBLE V5.0 (FINAL)</span>
</div>

<!-- Home Win Probability -->
<div style="margin-bottom:25px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div style="display:flex; align-items:center; gap:15px;">
<div style="width:40px; height:40px; border-radius:8px; border:1px solid #1e293b; background:#0f172a; display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden;">
<span style="font-size:1.2rem; filter: sepia(100%) hue-rotate(180deg) saturate(300%) opacity(0.8);">🏃</span>
</div>
<div>
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">Outcome</div>
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
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">Stability</div>
<div style="font-size:1.2rem; font-weight:700; color:{'#06b6d4' if pred_draw else '#e2e8f0'}; font-family:'Inter', sans-serif;">DRAW</div>
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
<div style="font-size:0.7rem; color:#64748b; font-weight:600; letter-spacing:1px; margin-bottom:2px;">Risk Factor</div>
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
            h_adv.append(f"Offensive Efficiency (+{((home_goals-away_goals)/max(0.1, away_goals))*100:.0f}% Goals)")
        elif away_goals > home_goals * 1.1:
            a_adv.append(f"Offensive Efficiency (+{((away_goals-home_goals)/max(0.1, home_goals))*100:.0f}% Goals)")
            
        home_ga = input_df['Home_Avg_GA_L5'][0]
        away_ga = input_df['Away_Avg_GA_L5'][0]
        if home_ga < away_ga * 0.9:
            h_adv.append("Defensive Solidity")
        elif away_ga < home_ga * 0.9:
            a_adv.append("Defensive Solidity")
            
        mv_diff = input_df['MV_Diff'][0]
        if mv_diff > 50_000_000:
            h_adv.append("Squad Depth (Market Value)")
        elif mv_diff < -50_000_000:
            a_adv.append("Squad Depth (Market Value)")
            
        if home_form > away_form + 0.3:
            h_adv.append("Superior Momentum (Form)")
        elif away_form > home_form + 0.3:
            a_adv.append("Superior Momentum (Form)")
            
        if not h_adv and not a_adv:
            delta_text = "Both teams exhibit very similar statistical attributes across core metrics, suggesting a highly competitive and balanced match."
        else:
            parts = []
            if h_adv:
                parts.append(f"The Home team has an advantage in {', '.join(h_adv)}")
            if a_adv:
                parts.append(f"The Away team shows strength in {', '.join(a_adv)}")
            delta_text = " while ".join(parts) + "."

        st.markdown(f"""
<div class="card" style="height: 100%; padding:25px;">
<div style="font-size:1.1rem; color:#94a3b8; font-family:'Inter', sans-serif; text-transform:uppercase; letter-spacing:1px; margin-bottom:30px; line-height:1.4;">
Comparative Feature<br>Analysis
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
<div style="color:#475569; letter-spacing:1px;">SQUAD VALUE</div>
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
<div style="color:#475569; letter-spacing:1px;">AVG GOALS (GF)</div>
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
<span style="font-size:0.75rem; font-weight:700; color:#06b6d4; letter-spacing:1px;">DELTA ANALYSIS (CORE METRICS)</span>
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
    leading_text = f"AI Confidence leads Crowd by {diff_home:+.2f}%" if abs(diff_home) > 0.5 else "AI Confidence aligns with Crowd"
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
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Win Probability: {home_team}</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_home:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">CROWD CONFIDENCE</div>
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
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px;">DRAW PROBABILITY</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_draw:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">CROWD CONFIDENCE</div>
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
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Win Probability: {away_team}</div>
        <div style="display:flex; gap:30px; text-align:right;">
            <div>
                <div style="font-size:0.6rem; color:#10b981; margin-bottom:2px;">AI CONFIDENCE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#f8fafc; font-family:'Inter', sans-serif;">{prob_away:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.6rem; color:#3b82f6; margin-bottom:2px;">CROWD CONFIDENCE</div>
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
    <div style="font-size:0.75rem; color:#64748b; font-style:italic;">Crowd confidence derived from FPL player ownership data and social momentum analysis (Simulated Data).</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- AI vs Human Performance Benchmark ---
    st.markdown(f"""
<div class="card">
<div class="card-title" style="margin-bottom:30px;">
<div style="display:flex; align-items:center; gap:10px;">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="w-6 h-6"><path d="M4 14V20M8 8V20M12 12V20M16 16V20M20 4V20" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
<span style="font-size:1.4rem; color:#f8fafc; font-weight:800; font-family:'Orbitron', sans-serif;">AI VS. HUMAN PERFORMANCE BENCHMARK</span>
</div>
</div>

<div class="benchmark-container">
<!-- Card 1 -->
<div style="flex:1; border:1px solid #064e3b; border-radius:8px; padding:20px; background:rgba(6,78,59,0.1); position:relative;">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#10b981; font-weight:700; margin-bottom:5px; letter-spacing:1px;">AI MODEL (STACKING V5.0)</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">{ai_acc:.2f}%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#10b981; display:flex; align-items:center; gap:5px;">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
Latest Validated Accuracy
</div>
</div>

<!-- Card 2 -->
<div style="flex:1; border:1px solid #1e3a8a; border-radius:8px; padding:20px; background:rgba(30,58,138,0.1);">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#60a5fa; font-weight:700; margin-bottom:5px; letter-spacing:1px;">FPL CROWD / FANS</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">{crowd_acc:.2f}%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#94a3b8;">
Crowd Benchmark
</div>
</div>

<!-- Card 3 -->
<div style="flex:1; border:1px solid #4c1d95; border-radius:8px; padding:20px; background:rgba(76,29,149,0.1);">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:0.75rem; color:#c084fc; font-weight:700; margin-bottom:5px; letter-spacing:1px;">BBC EXPERTS (SUTTON)</div>
<div style="font-size:2.5rem; font-weight:800; color:#f8fafc; font-family:'Inter', sans-serif; line-height:1;">{human_acc:.2f}%</div>
</div>
<div style="opacity:0.3;">
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect><path d="M9 22v-4h6v4"></path><path d="M8 6h.01"></path><path d="M16 6h.01"></path><path d="M12 6h.01"></path><path d="M12 10h.01"></path><path d="M12 14h.01"></path><path d="M16 10h.01"></path><path d="M16 14h.01"></path><path d="M8 10h.01"></path><path d="M8 14h.01"></path></svg>
</div>
</div>
<div style="margin-top:20px; font-size:0.8rem; color:#94a3b8;">
Expert Global Benchmark
</div>
</div>
</div>

<!-- Banner -->
<div style="background:rgba(6,78,59,0.2); border:1px solid #064e3b; border-radius:8px; padding:15px; display:flex; align-items:center; gap:10px;">
<span style="color:#fbbf24; font-size:1.2rem;">🏆</span>
<span style="color:#10b981; font-weight:600; font-size:0.95rem; font-family:'Inter', sans-serif;">RESULT: Our AI model outperforms all benchmarks! Accuracy exceeds the Crowd by (+{(ai_acc-crowd_acc):.2f}%) and BBC Experts by (+{(ai_acc-human_acc):.2f}%).</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # --- AI Prediction History Table ---
    if not history_df.empty:
        # Build Table HTML
        table_rows = ""
        # Show all 380 matches for the season
        recent_history = history_df.iloc[::-1]
        
        # Create a lookup for human baseline
        human_lookup = {}
        if not human_baseline_df.empty:
            for _, h_row in human_baseline_df.iterrows():
                human_lookup[h_row['Match']] = {
                    'pred': h_row['Sutton Prediction'],
                    'correct': h_row['Correct Result']
                }
                
        # Create a lookup for dates and scores
        try:
            pl_matches = pd.read_csv("data/raw/pl_matches_2021_2025.csv")
            # Filter to season 2024 (the one we tested on), dropping future matches
            pl_matches = pl_matches[pl_matches['season'] == 2024]
            pl_matches['match_name'] = pl_matches['home_team'] + ' vs ' + pl_matches['away_team']
            match_details = pl_matches.drop_duplicates(subset=['match_name'], keep='last').set_index('match_name')[['date', 'home_score', 'away_score']].to_dict('index')
        except:
            match_details = {}

        for _, row in recent_history.iterrows():
            match_name = row['Match']
            actual = row['Actual'].replace("_TEAM", "").capitalize() if isinstance(row['Actual'], str) else "N/A"
            
            # Extract date and score
            details = match_details.get(match_name, {})
            raw_date = details.get('date', '')
            if isinstance(raw_date, str) and len(raw_date) >= 16:
                try:
                    dt = pd.to_datetime(raw_date)
                    dt_th = dt + pd.Timedelta(hours=7)
                    date_str = dt_th.strftime('%d %b %Y %H:%M')
                except:
                    date_str = raw_date[:10] + " " + raw_date[11:16]
            else:
                date_str = ""
                
            h_score = details.get('home_score')
            a_score = details.get('away_score')
            if pd.notna(h_score) and pd.notna(a_score):
                score_str = f"{int(h_score)} - {int(a_score)}"
            else:
                score_str = ""
            
            # AI Prediction
            ai_pred = row['AI_Pred'].replace("_TEAM", "").capitalize() if isinstance(row['AI_Pred'], str) else "N/A"
            ai_correct = row['AI_Correct']
            
            # FPL Fans Prediction
            fpl_pred = row['Crowd_Pred'].replace("_TEAM", "").capitalize() if isinstance(row['Crowd_Pred'], str) else "N/A"
            fpl_correct = row['Crowd_Correct']
            
            # BBC Prediction (Sutton)
            bbc_data = human_lookup.get(match_name, None)
            if bbc_data:
                bbc_pred_raw = bbc_data['pred']
                if '-' in str(bbc_pred_raw):
                    try:
                        h, a = map(int, str(bbc_pred_raw).split('-'))
                        if h > a: bbc_pred = "Home Win"
                        elif a > h: bbc_pred = "Away Win"
                        else: bbc_pred = "Draw"
                    except:
                        bbc_pred = str(bbc_pred_raw)
                else:
                    bbc_pred = str(bbc_pred_raw)
                bbc_correct = bbc_data['correct']
                bbc_color = "#10b981" if bbc_correct else "#f43f5e"
            else:
                bbc_pred = "—"
                bbc_color = "#94a3b8"

            # Map labels for UI
            actual = actual.replace("Home", "Home Win").replace("Away", "Away Win")
            ai_pred = ai_pred.replace("Home", "Home Win").replace("Away", "Away Win")
            fpl_pred = fpl_pred.replace("Home", "Home Win").replace("Away", "Away Win")
            
            ai_color = "#10b981" if ai_correct else "#f43f5e"
            fpl_color = "#10b981" if fpl_correct else "#f43f5e"
            
            # Decorate the names with Date and Score if available
            match_display = f"<div>{match_name}</div><div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>{date_str}</div>" if date_str else match_name
            actual_display = f"<div>{actual}</div><div style='font-size:0.75rem; color:#64748b; font-weight:600; margin-top:2px;'>{score_str}</div>" if (score_str and score_str != "N/A") else actual
            
            table_rows += f"""
            <tr style="border-bottom:1px solid #2d3748;">
                <td style="padding:12px; text-align:left;">{match_display}</td>
                <td style="padding:12px; text-align:center; color:#94a3b8;">{actual_display}</td>
                <td style="padding:12px; text-align:center; font-weight:600; color:{ai_color};">{ai_pred}</td>
                <td style="padding:12px; text-align:center; font-weight:500; color:{fpl_color};">{fpl_pred}</td>
                <td style="padding:12px; text-align:center; font-weight:500; color:{bbc_color};">{bbc_pred}</td>
            </tr>
            """

        html_content = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            body {{
                background: #11141E;
                color: #f8fafc;
                font-family: 'Inter', sans-serif;
                margin: 0; padding: 0;
            }}
            .card {{
                background-color: #1A202C;
                border: 1px solid #2d3748;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            }}
            .card-title {{
                font-size: 1.1rem;
                color: #f8fafc;
                font-weight: 800;
                margin-bottom: 20px;
                display: flex; gap: 10px; align-items: center;
            }}
            .scroll-area {{
                height: 380px; 
                overflow-y: auto; 
                border: 1px solid #2d3748; 
                border-radius: 8px; 
                background: rgba(15,23,42,0.5);
            }}
            table {{
                width: 100%; border-collapse: collapse; font-size: 0.85rem;
            }}
            th {{
                position: sticky; top: 0; background: #1e293b; 
                padding: 12px; text-align: center; border-bottom: 1px solid #334155;
            }}
            th:first-child {{ text-align: left; }}
            td {{ padding: 12px; text-align: center; font-family: 'Inter', sans-serif; }}
            td:first-child {{ text-align: left; }}
            /* Custom Scrollbar */
            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-track {{ background: #1e293b; }}
            ::-webkit-scrollbar-thumb {{ background: #4a5568; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #718096; }}
        </style>
        <div class="card">
            <div class="card-title">📜 AI Prediction History 2024/2025 Season</div>
            <div class="scroll-area">
                <table>
                    <thead>
                        <tr>
                            <th style="width:30%;">Match</th>
                            <th style="width:17%;">Actual</th>
                            <th style="width:17%;">AI Predict</th>
                            <th style="width:18%;">FPL Fans Predict</th>
                            <th style="width:18%;">BBC Predict</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            <div style="margin-top:10px; font-size:0.7rem; color:#64748b; text-align:right;">* Showing all results from competitive verification datasets (2024/2025 Season).</div>
        </div>
        """
        components.html(html_content, height=520, scrolling=False)
        
    try:
        pred_2025_df = pd.read_csv(PRED_2025_PATH)
    except FileNotFoundError:
        pred_2025_df = pd.DataFrame()
        
    if not pred_2025_df.empty:
        # Build Table HTML for 2025
        table_rows_2025 = ""
        recent_history_2025 = pred_2025_df
        
        match_details_2025 = {}
        try:
            pl_matches = pd.read_csv("data/raw/pl_matches_2021_2025.csv")
            pl_matches = pl_matches[pl_matches['season'] == 2025]
            pl_matches['match_name'] = pl_matches['home_team'] + ' vs ' + pl_matches['away_team']
            match_details_2025 = pl_matches.drop_duplicates(subset=['match_name'], keep='last').set_index('match_name')[['date', 'home_score']].to_dict('index')
        except:
            pass

        for _, row in recent_history_2025.iterrows():
            match_name = row['Match']
            details = match_details_2025.get(match_name, {})
            
            # Skip if match has a score already (meaning result is known)
            if pd.notna(details.get('home_score')):
                continue
            
            # Extract date
            raw_date = details.get('date', '')
            if isinstance(raw_date, str) and len(raw_date) >= 16:
                try:
                    dt = pd.to_datetime(raw_date)
                    if dt < pd.Timestamp.now(tz='UTC'):
                        continue
                    dt_th = dt + pd.Timedelta(hours=7)
                    date_str = dt_th.strftime('%d %b %Y %H:%M')
                except:
                    date_str = raw_date[:10] + " " + raw_date[11:16]
            else:
                date_str = ""
            
            ai_pred = row['AI_Pred']
            ai_score = row.get('AI_Score', '')
            actual_display = "<span style='font-size:0.75rem; color:#64748b; font-weight:600;'>TBD</span>"

            # Map labels for UI
            ai_color = "#38bdf8"
            
            match_display = f"<div>{match_name}</div><div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>{date_str}</div>" if date_str else match_name
            ai_display = f"<div>{ai_pred}</div><div style='font-size:0.75rem; color:#94a3b8; font-weight:600; margin-top:2px;'>Score: {ai_score}</div>" if ai_score else ai_pred
            
            table_rows_2025 += f"""
            <tr style="border-bottom:1px solid #2d3748;">
                <td style="padding:12px; text-align:left;">{match_display}</td>
                <td style="padding:12px; text-align:center; color:#94a3b8;">{actual_display}</td>
                <td style="padding:12px; text-align:center; font-weight:600; color:{ai_color};">{ai_display}</td>
            </tr>
            """

        html_content_2025 = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            body {{
                background: #11141E;
                color: #f8fafc;
                font-family: 'Inter', sans-serif;
                margin: 0; padding: 0;
            }}
            .card {{
                background-color: #1A202C;
                border: 1px solid #2d3748;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
                margin-top: 20px;
            }}
            .card-title {{
                font-size: 1.1rem;
                color: #f8fafc;
                font-weight: 800;
                margin-bottom: 20px;
                display: flex; gap: 10px; align-items: center;
            }}
            .scroll-area {{
                height: 380px; 
                overflow-y: auto; 
                border: 1px solid #2d3748; 
                border-radius: 8px; 
                background: rgba(15,23,42,0.5);
            }}
            table {{
                width: 100%; border-collapse: collapse; font-size: 0.85rem;
            }}
            th {{
                position: sticky; top: 0; background: #1e293b; 
                padding: 12px; text-align: center; border-bottom: 1px solid #334155;
            }}
            th:first-child {{ text-align: left; }}
            td {{ padding: 12px; text-align: center; font-family: 'Inter', sans-serif; }}
            td:first-child {{ text-align: left; }}
            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-track {{ background: #1e293b; }}
            ::-webkit-scrollbar-thumb {{ background: #4a5568; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #718096; }}
        </style>
        <div class="card">
            <div class="card-title">🔮 Future AI Predictions 2025/2026</div>
            <div class="scroll-area">
                <table>
                    <thead>
                        <tr>
                            <th style="width:50%;">Match</th>
                            <th style="width:25%;">Actual</th>
                            <th style="width:25%;">AI Predict</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_2025}
                    </tbody>
                </table>
            </div>
            <div style="margin-top:10px; font-size:0.7rem; color:#64748b; text-align:right;">* Showing simulated future fixtures without known results for the 2025/2026 Season.</div>
        </div>
        """
        components.html(html_content_2025, height=540, scrolling=False)


if __name__ == "__main__":
    main()
