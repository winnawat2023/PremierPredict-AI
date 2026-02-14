import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(page_title="PremierPredict-AI", layout="wide")

# Load Resources
@st.cache_resource
def load_resources():
    model = joblib.load("models/rf_model.pkl")
    try:
        metrics = json.load(open("models/metrics.json"))
    except FileNotFoundError:
        metrics = {"baseline_accuracy": 0, "ai_accuracy": 0}
        
    try:
        feature_imp = pd.read_csv("models/feature_importance.csv")
    except FileNotFoundError:
        feature_imp = pd.DataFrame(columns=["Feature", "Importance"])
        
    return model, metrics, feature_imp

def main():
    st.title("⚽ PremierPredict-AI: Premier League Match Predictor")
    st.markdown("### SEA612 Artificial Intelligence Fundamentals")
    
    # Load resources
    if not os.path.exists("models/rf_model.pkl"):
        st.error("Model not found. Please run `python src/models.py` first.")
        return

    model, metrics, feature_imp = load_resources()
    
    # Sidebar
    st.sidebar.header("Match Outcome Prediction")
    st.sidebar.markdown("Adjust features to predict:")
    
    home_adv = st.sidebar.selectbox("Home Advantage", [0, 1], index=1, help="1 if playing at Home, 0 if Neutral/Away context (simplified)")
    home_form = st.sidebar.slider("Home Team Form (Last 5 Avg Points)", 0.0, 3.0, 1.5, 0.1)
    away_form = st.sidebar.slider("Away Team Form (Last 5 Avg Points)", 0.0, 3.0, 1.5, 0.1)
    pos_diff = st.sidebar.number_input("Position Difference (Home Rank - Away Rank)", min_value=-20, max_value=20, value=0, help="Negative means Home is higher ranked (better).")
    
    # Prediction
    input_data = pd.DataFrame({
        'Home_Advantage': [home_adv],
        'Home_Form_L5': [home_form],
        'Away_Form_L5': [away_form],
        'Position_Diff': [pos_diff]
    })
    
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        
        # Outcome Map
        outcomes = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        result = outcomes[prediction]
        
        st.subheader("Prediction Result")
        st.info(f"The AI predicts: **{result}**")
        
        # Probability Bar
        st.markdown("#### Probability Distribution")
        prob_df = pd.DataFrame({
            "Outcome": ["Home Win", "Draw", "Away Win"],
            "Probability": probs
        })
        
        # Display as 3-color bar chart or columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{probs[0]*100:.1f}%")
        col2.metric("Draw", f"{probs[1]*100:.1f}%")
        col3.metric("Away Win", f"{probs[2]*100:.1f}%")
        
        # Simple Bar Chart
        fig, ax = plt.subplots(figsize=(6, 2))
        sns.barplot(x="Probability", y="Outcome", data=prob_df, ax=ax, palette=["green", "gray", "red"])
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    # Dashboard Comparison
    st.divider()
    st.subheader("📊 Model Performance: AI vs Baseline")
    
    comp_col1, comp_col2 = st.columns(2)
    
    baseline_acc = metrics['baseline_accuracy'] * 100
    ai_acc = metrics['ai_accuracy'] * 100
    
    comp_col1.metric("Baseline Accuracy (Home Win Strategy)", f"{baseline_acc:.2f}%")
    comp_col2.metric("AI Model Accuracy (Random Forest)", f"{ai_acc:.2f}%", delta=f"{ai_acc - baseline_acc:.2f}%")
    
    if ai_acc > baseline_acc:
        st.success("✅ Goal Achieved: AI outperforms Baseline.")
    else:
        st.warning("⚠️ Research Finding: AI performs similar to or below Baseline.")
        
    # Feature Importance
    st.subheader("🔍 Feature Importance (Explainable AI)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=feature_imp, ax=ax2, palette="viridis")
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
