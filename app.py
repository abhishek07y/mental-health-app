from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ================== EDIT 1: model file name ==================
MODEL_FILE = "Mental_Heath_Model.pkl"   # keep it in the same folder as app.py
# =============================================================

st.set_page_config(page_title="Mental Health Score", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .big-score {font-size: 3.2rem; font-weight: 700; line-height: 1;}
    .sub {color: #6b7280; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Model load ----------
model_path = Path(__file__).parent / MODEL_FILE
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(model_path)


model = load_model()

# ---------- Header ----------
st.title("🧠 Mental Health Score Predictor")
st.markdown(
    '<p class="sub">Enter your social media usage and lifestyle details, and the model will estimate your mental health score.</p>',
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("👤 Personal info")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 10, 60, 20)
        gender = c2.selectbox("Gender", ["Female", "Male"])
        academic = st.selectbox("Academic level", ["High School", "Undergraduate", "Graduate"])
        country = st.selectbox(
            "Country",
            ["India", "USA", "UK", "Canada", "Australia", "France",
             "Germany", "Mexico", "Turkey", "Other"],
        )

    with st.container(border=True):
        st.subheader("📱 Social media")
        platform = st.selectbox(
            "Most used platform",
            ["Instagram", "Facebook", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
             "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube"],
        )
        purpose = st.selectbox("Purpose of use", ["Education", "Entertainment", "Networking", "News"])
        usage = st.slider("Daily usage (hours)", 0.0, 16.0, 5.0, 0.5)
        unlocks = st.slider("Daily phone unlocks", 0, 400, 150, 5)

with right:
    with st.container(border=True):
        st.subheader("🏃 Lifestyle")
        study = st.slider("Study hours (per day)", 0.0, 16.0, 3.0, 0.5)
        activity = st.slider("Physical activity (hours)", 0.0, 6.0, 1.5, 0.25)
        sleep = st.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.5)
        stress = st.select_slider(
            "Stress level", ["Low", "Medium", "High", "Very High"], value="Medium"
        )

    predict = st.button("🔍 Get my score", type="primary", use_container_width=True)

    # ---------- Result ----------
    if predict:
        data = pd.DataFrame([{
            "Study_Hours": study,
            "Age": age,
            "Avg_Daily_Usage_Hours": usage,
            "Daily_Unlocks": unlocks,
            "Physical_Activity_Hours": activity,
            "Sleep_Hours_Per_Night": sleep,
            "Stress_Level": stress,
            "Gender": gender,
            "Academic_Level": academic,
            "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose,
            "grouped_country": country,
        }])
        score = float(model.predict(data)[0])

        with st.container(border=True):
            st.subheader("📊 Result")
            st.markdown(
                f'<div class="big-score">{score:.2f}<span class="sub"> / 10</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score / 10, 0.0), 1.0))

            # Bands are an assumption (higher = better). Adjust to match your dataset.
            if score >= 7:
                st.success("Good score 👍 Your lifestyle looks balanced.")
            elif score >= 5:
                st.warning("Medium score. Pay attention to your sleep, activity and screen time.")
            else:
                st.error("Low score. Try to reduce screen time and improve your sleep and activity.")

            tips = []
            if sleep < 7:
                tips.append("Try to get around 7 hours of sleep.")
            if usage > 6:
                tips.append("Try to keep daily screen time under 6 hours.")
            if activity < 1:
                tips.append("Get at least 1 hour of physical activity every day.")
            if stress in ("High", "Very High"):
                tips.append("Your stress is high. Talk to someone you trust or take a break.")
            if tips:
                st.markdown("**Suggestions**")
                for t in tips:
                    st.markdown(f"- {t}")

st.caption("⚠️ This is only an ML estimate, not a medical diagnosis.")