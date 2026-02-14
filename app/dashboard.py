import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_latest_team_stats

# Set page config
st.set_page_config(page_title="PremierPredict-AI", layout="wide")

# Load Resources
@st.experimental_singleton
def load_resources():
    try:
        model = joblib.load("models/rf_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    try:
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {"baseline_accuracy": 0, "ai_accuracy": 0}
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        metrics = {"baseline_accuracy": 0, "ai_accuracy": 0}
        
    try:
        feature_imp = pd.read_csv("models/feature_importance.csv")
    except FileNotFoundError:
        feature_imp = pd.DataFrame(columns=["Feature", "Importance"])
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
        feature_imp = pd.DataFrame(columns=["Feature", "Importance"])
        
    return model, metrics, feature_imp

def main():
    st.title("⚽ PremierPredict-AI: ระบบทำนายผลฟุตบอลพรีเมียร์ลีก")
    st.markdown("### SEA612 พื้นฐานปัญญาประดิษฐ์ (Artificial Intelligence Fundamentals)")
    
    # Load resources
    if not os.path.exists("models/rf_model.pkl"):
        st.error("ไม่พบโมเดล กรุณารัน `python src/models.py` ก่อน")
        return

    model, metrics, feature_imp = load_resources()
    
    if model is None:
        st.error("Failed to load model. Please check the logs.")
        return
    
    # Sidebar
    st.sidebar.header("ทำนายผลการแข่งขัน")
    st.sidebar.markdown("ปรับค่าตัวแปรเพื่อทำนาย:")
    
    # Mode Selection
    mode = st.sidebar.radio("โหมดข้อมูล", ["เลือกทีมแข่งขัน", "ระบุค่าเอง (Manual)"])
    
    if mode == "เลือกทีมแข่งขัน":
        team_stats = get_latest_team_stats()
        team_names = sorted(team_stats.keys())
        
        home_team = st.sidebar.selectbox("ทีมเจ้าบ้าน", team_names, index=0)
        away_team = st.sidebar.selectbox("ทีมเยือน", team_names, index=1)
        
        if home_team == away_team:
            st.sidebar.error("ทีมเจ้าบ้านและทีมเยือนต้องไม่ซ้ำกัน")
        
        # Auto-fill features
        h_stats = team_stats.get(home_team, {'rank': 10, 'form': 1.5})
        a_stats = team_stats.get(away_team, {'rank': 10, 'form': 1.5})
        
        home_adv = 1 # Always 1 for home team
        home_form = h_stats['form']
        away_form = a_stats['form']
        pos_diff = h_stats['rank'] - a_stats['rank']
        
        # Display calculated values
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**ค่าที่คำนวณอัตโนมัติ:**")
        st.sidebar.text(f"ฟอร์มเจ้าบ้าน (5 นัดหลัง): {home_form:.2f}")
        st.sidebar.text(f"ฟอร์มทีมเยือน (5 นัดหลัง): {away_form:.2f}")
        st.sidebar.text(f"ผลต่างอันดับ: {pos_diff} (อันดับ {h_stats['rank']} vs {a_stats['rank']})")
        
    else:
        home_adv = st.sidebar.selectbox("ความได้เปรียบเจ้าบ้าน", [0, 1], index=1, help="1 ถ้าเล่นในบ้าน, 0 ถ้าสนามกลาง/เยือน")
        home_form = st.sidebar.slider("ฟอร์มเจ้าบ้าน (คะแนนเฉลี่ย 5 นัดหลัง)", 0.0, 3.0, 1.5, 0.1)
        away_form = st.sidebar.slider("ฟอร์มทีมเยือน (คะแนนเฉลี่ย 5 นัดหลัง)", 0.0, 3.0, 1.5, 0.1)
        pos_diff = st.sidebar.number_input("ผลต่างอันดับ (อันดับเจ้าบ้าน - อันดับทีมเยือน)", min_value=-20, max_value=20, value=0, help="ค่าลบหมายถึงเจ้าบ้านอันดับดีกว่า")
    
    # Prediction
    input_data = pd.DataFrame({
        'Home_Advantage': [home_adv],
        'Home_Form_L5': [home_form],
        'Away_Form_L5': [away_form],
        'Position_Diff': [pos_diff]
    })
    
    # Prediction Logic (Reactive)
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    
    # Outcome Map
    outcomes = {0: "เจ้าบ้านชนะ (Home Win)", 1: "เสมอ (Draw)", 2: "ทีมเยือนชนะ (Away Win)"}
    result = outcomes[prediction]
    
    st.subheader("ผลการทำนาย")
    st.info(f"AI ทำนายว่า: **{result}**")
    
    # Probability Bar
    st.markdown("#### ความน่าจะเป็น (Probability)")
    
    # Data for Plotting (Use English to avoid font issues)
    prob_df_plot = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Probability": probs
    })
    
    # Display as 3-color bar chart or columns (Thai Text)
    col1, col2, col3 = st.columns(3)
    col1.metric("เจ้าบ้านชนะ (Home)", f"{probs[0]*100:.1f}%")
    col2.metric("เสมอ (Draw)", f"{probs[1]*100:.1f}%")
    col3.metric("ทีมเยือนชนะ (Away)", f"{probs[2]*100:.1f}%")
    
    # Simple Bar Chart
    fig, ax = plt.subplots(figsize=(6, 2))
    sns.barplot(x="Probability", y="Outcome", hue="Outcome", data=prob_df_plot, ax=ax, palette=["green", "gray", "red"], legend=False)
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    # Dashboard Comparison
    st.markdown("---")
    st.subheader("📊 ประสิทธิภาพของโมเดล (Model Performance)")
    
    col_b1, col_b2, col_b3, col_ai = st.columns(4)
    
    base1_acc = metrics.get('baseline1_accuracy', 0) * 100
    base2_acc = metrics.get('baseline2_accuracy', 0) * 100
    base3_acc = metrics.get('baseline3_accuracy', 0) * 100
    ai_acc = metrics.get('ai_accuracy', 0) * 100
    
    col_b1.metric("Baseline 1 (Home Win)", f"{base1_acc:.2f}%", help="ทายเจ้าบ้านชนะตลอด")
    col_b2.metric("Baseline 2 (Rank)", f"{base2_acc:.2f}%", help="ทายตามอันดับตารางคะแนน")
    col_b3.metric("Baseline 3 (Random)", f"{base3_acc:.2f}%", help="สุ่มตามความน่าจะเป็น")
    col_ai.metric("AI Model (Random Forest)", f"{ai_acc:.2f}%", delta=f"{ai_acc - base2_acc:.2f}% vs Rank")
    
    if ai_acc > base2_acc:
        st.success("✅ เป้าหมายสำเร็จ: AI ชนะทุก Baseline")
    elif ai_acc > base1_acc:
        st.warning("⚠️ สำเร็จบางส่วน: AI ชนะกลยุทธ์เจ้าบ้าน แต่ยังแพ้กลยุทธ์อันดับ")
    else:
        st.error("❌ ความท้าทาย: AI ยังแม่นยำน้อยกว่า Baseline ง่ายๆ")
        
    # Feature Importance
    st.subheader("🔍 ความสำคัญของตัวแปร (Feature Importance)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # Use English for Feature Names if they aren't already
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=feature_imp, ax=ax2, palette="viridis", legend=False)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
