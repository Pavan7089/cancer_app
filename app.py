import streamlit as st
import pandas as pd
import joblib
import json

# --------------------------
# LOAD MODEL & ENCODER
# --------------------------
model = joblib.load("rf_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
feature_cols = json.load(open("feature_columns.json", "r"))

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Cancer Level Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------
# CUSTOM CSS (clean, minimal)
# --------------------------
st.markdown("""
<style>
    .main { background-color: #f4f7fb; }
    .card {
        background: white;
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .title-text {
        font-size: 28px;
        font-weight: 600;
        color: #1a3d6d;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle-text {
        color: #4d4d4d;
        font-size: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-low { color: #1b8f3a; font-size: 24px; font-weight: 600; }
    .result-medium { color: #e6a700; font-size: 24px; font-weight: 600; }
    .result-high { color: #c0392b; font-size: 24px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown("<div class='title-text'>Cancer Risk Level Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Provide the patient details below to estimate cancer risk.</div>", unsafe_allow_html=True)

# --------------------------
# INPUT FORM CARD
# --------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, value=45)
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        gender = 1 if gender_label == "Male" else 0
        air_pollution = st.slider("Air Pollution", 1, 10, 5)
        alcohol_use = st.slider("Alcohol use", 1, 10, 5)

    with col2:
        smoking = st.slider("Smoking", 1, 10, 5)
        coughing_blood = st.slider("Coughing of Blood", 1, 10, 5)
        fatigue = st.slider("Fatigue", 1, 10, 5)
        sob = st.slider("Shortness of Breath", 1, 10, 5)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# PREDICTION BUTTON
# --------------------------
if st.button("Predict Risk Level"):
    try:
        inputs = {
            "Age": age,
            "Gender": gender,
            "Air Pollution": air_pollution,
            "Alcohol use": alcohol_use,
            "Smoking": smoking,
            "Coughing of Blood": coughing_blood,
            "Fatigue": fatigue,
            "Shortness of Breath": sob
        }

        df = pd.DataFrame([[inputs[col] for col in feature_cols]], columns=feature_cols)

        pred_encoded = model.predict(df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0].lower()

        # Risk styling
        if pred_label == "low":
            style = "result-low"
            text = "Low Risk"
        elif pred_label == "medium":
            style = "result-medium"
            text = "Medium Risk"
        else:
            style = "result-high"
            text = "High Risk"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='{style}'>{text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Error during prediction.")
        st.write(str(e))
else:
    st.info("Fill the form and click **Predict Risk Level**.")
