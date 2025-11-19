# app.py — Streamlit cancer risk prediction (Corrected for 8-feature RF model)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# =============== LOAD FEATURE COLUMNS ===============

with open("feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

# Expected:
# ["Age", "Gender", "Air Pollution", "Alcohol use",
#  "Smoking", "Coughing of Blood", "Fatigue", "Shortness of Breath"]

MODEL_PATH = "rf_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

# =============== LOAD MODEL ===============

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

model, label_encoder = load_artifacts()

# =============== PAGE CONFIG ===============

st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("Cancer Risk Prediction Dashboard")
st.write("Fill the patient details below to estimate cancer risk using a Random Forest model.")

# =============== INPUT FORM ===============

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        air_pollution = st.slider("Air Pollution (1-10)", 1, 10, 5)
        alcohol_use = st.slider("Alcohol use (1-10)", 1, 10, 5)

    with col2:
        smoking = st.slider("Smoking (1-10)", 1, 10, 5)
        coughing_blood = st.slider("Coughing of Blood (1-10)", 1, 10, 5)
        fatigue = st.slider("Fatigue (1-10)", 1, 10, 5)
        sob = st.slider("Shortness of Breath (1-10)", 1, 10, 5)

    submitted = st.form_submit_button("Predict Risk")

# Build input with ONLY the 8 trained features
input_data = {
    "Age": age,
    "Gender": gender,
    "Air Pollution": air_pollution,
    "Alcohol use": alcohol_use,
    "Smoking": smoking,
    "Coughing of Blood": coughing_blood,
    "Fatigue": fatigue,
    "Shortness of Breath": sob
}

df_input = pd.DataFrame([input_data])
df_input = df_input[FEATURE_COLUMNS]  # EXACT match to model training

# =============== PREDICTION LOGIC ===============

if submitted:
    try:
        pred_encoded = model.predict(df_input)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.subheader("Predicted Risk Level")
        st.write(f"**Risk:** {pred_label}")

        # Optional: Probability table
        try:
            probs = model.predict_proba(df_input)[0]
            class_labels = label_encoder.inverse_transform(np.arange(len(probs)))
            prob_df = pd.DataFrame({"Level": class_labels, "Probability": probs})
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x*100:.2f}%")
            st.write("Prediction probabilities:")
            st.dataframe(prob_df, hide_index=True)
        except:
            pass

    except Exception as e:
        st.error("Prediction failed. Check logs and ensure input matches the trained model.")
        st.write(e)
else:
    st.info("Enter details and click Predict.")
