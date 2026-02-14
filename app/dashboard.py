import streamlit as st
import os
import sys

# 1. Page Config (First command)
st.set_page_config(page_title="PremierPredict-AI Debug", layout="wide")

st.title("🚧 Deep Debug Mode")
st.markdown("If you see this, Streamlit itself is working!")

# 2. Environment Info
st.subheader("1. Environment Info")
st.write(f"**CWD:** `{os.getcwd()}`")
st.write(f"**Python:** `{sys.version}`")

# 3. Import Check
st.subheader("2. Library Import Check")

libs = ["pandas", "joblib", "json", "matplotlib", "seaborn", "sklearn"]
for lib in libs:
    try:
        if lib == "matplotlib":
            import matplotlib
            matplotlib.use('Agg') # Force Agg before importing pyplot
            import matplotlib.pyplot as plt
        elif lib == "sklearn":
            import sklearn
            st.write(f"✅ `{lib}` imported (v{sklearn.__version__})")
            continue
        else:
            __import__(lib)
        st.write(f"✅ `{lib}` imported")
    except Exception as e:
        st.error(f"❌ Failed to import `{lib}`: {e}")

# 4. Project Module Check
st.subheader("3. Project Module Check")
try:
    # Add root to path
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(root_path)
    st.write(f"Added to sys.path: `{root_path}`")
    
    from src.utils import get_latest_team_stats
    st.write("✅ `src.utils` imported successfully")
except Exception as e:
    st.error(f"❌ Failed to import project modules: {e}")

# 5. Model Loading Check
st.subheader("4. Model Loading Check")
try:
    import joblib
    model_path = "models/rf_model.pkl"
    if os.path.exists(model_path):
        st.write(f"Found model file at `{model_path}`")
        model = joblib.load(model_path)
        st.write(f"✅ Model loaded successfully: `{type(model)}`")
    else:
        st.error(f"❌ Model file not found at `{model_path}`")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

st.success("Debug run complete.")
