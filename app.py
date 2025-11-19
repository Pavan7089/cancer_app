import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# Load Model & Encoder
# ----------------------------
model = joblib.load("models/model_rf.joblib")        # your trained RF model
label_encoder = joblib.load("models/label_encoder.joblib")

# ----------------------------
# Streamlit Page Settings
# ----------------------------
st.set_page_config(
    page_title="Cancer Risk Level Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Professional CSS Styling
# ----------------------------
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 38px;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        padding-bottom: 10px;
    }
    .sub {
        color: #34495E;
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
    }
    .report-box {
        padding: 20px;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    }
    .result {
        font-size: 26px;
        font-weight: 700;
        padding: 10px;
        color: #1B4F72;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.markdown("<div class='title'>🩺 Cancer Risk Level Prediction System</div>", unsafe_allow_html=True)
st.write("Provide patient health parameters below to estimate cancer risk level.")

# ---------------------------------------
# Input Form (modern two-column layout)
# ---------------------------------------
st.markdown("<div class='sub'>Patient Parameters</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

inputs = {}

with col1:
    inputs["Age"] = st.number_input("Age", 1, 120, 40)
    inputs["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    inputs["Air Pollution"] = st.slider("Air Pollution", 1, 10, 5)
    inputs["Alcohol use"] = st.slider("Alcohol Use", 1, 10, 5)
    inputs["Dust Allergy"] = st.slider("Dust Allergy", 1, 10, 5)
    inputs["Genetic Risk"] = st.slider("Genetic Risk", 1, 10, 5)
    inputs["Obesity"] = st.slider("Obesity", 1, 10, 5)

with col2:
    inputs["Smoking"] = st.slider("Smoking", 1, 10, 5)
    inputs["Passive Smoker"] = st.slider("Passive Smoker", 1, 10, 5)
    inputs["Chest Pain"] = st.slider("Chest Pain", 1, 10, 5)
    inputs["Coughing of Blood"] = st.slider("Coughing of Blood", 1, 10, 5)
    inputs["Fatigue"] = st.slider("Fatigue", 1, 10, 5)
    inputs["Shortness of Breath"] = st.slider("Shortness of Breath", 1, 10, 5)
    inputs["Dry Cough"] = st.slider("Dry Cough", 1, 10, 5)

with col3:
    inputs["Wheezing"] = st.slider("Wheezing", 1, 10, 5)
    inputs["Frequent Cold"] = st.slider("Frequent Cold", 1, 10, 5)
    inputs["Snoring"] = st.slider("Snoring", 1, 10, 5)
    inputs["Balanced Diet"] = st.slider("Balanced Diet", 1, 10, 5)
    inputs["Weight Loss"] = st.slider("Weight Loss", 1, 10, 5)
    inputs["Swallowing Difficulty"] = st.slider("Swallowing Difficulty", 1, 10, 5)
    inputs["Clubbing of Finger Nails"] = st.slider("Clubbing of Finger Nails", 1, 10, 5)

# Convert to DF
df_input = pd.DataFrame([inputs])

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict Risk Level", use_container_width=True):

    pred_encoded = model.predict(df_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Show Result Box
    st.markdown("<div class='sub'>Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>Predicted Cancer Risk Level: {pred_label}</div>", unsafe_allow_html=True)

    # ----------------------------
    # Generate Risk Report
    # ----------------------------
    st.markdown("<div class='sub'>📊 Personalized Risk Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='report-box'>", unsafe_allow_html=True)

    # Determine risky factors
    risky = {k: v for k, v in inputs.items() if v >= 7}
    good = {k: v for k, v in inputs.items() if v <= 3}

    st.write("### 🔴 High-Risk Factors (Need Immediate Attention)")
    if risky:
        for k, v in risky.items():
            st.write(f"**• {k}** — Value: {v}/10")
    else:
        st.write("No major high-risk indicators detected.")

    st.write("### 🟢 Protective Factors (Good Levels)")
    if good:
        for k, v in good.items():
            st.write(f"**• {k}** — Value: {v}/10")
    else:
        st.write("No protective factors detected.")

    # ----------------------------
    # Radar Chart (Spider Chart)
    # ----------------------------
    st.write("### 📈 Health Factor Insight Chart")
    fig = go.Figure()

    categories = list(inputs.keys())
    values = list(inputs.values())
    values.append(values[0])  # close loop

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Patient Score'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
