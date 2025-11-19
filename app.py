# app.py  — Streamlit cancer risk prediction (RF model, professional UI)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============== CONFIG ===============

# >>> ADJUST THESE PATHS IF NEEDED <<<
MODEL_PATH = "rf_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"


# These must match the columns used when training the RF model (order doesn’t matter, names must match)
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
]

# Factors where higher value = worse risk
HIGHER_IS_WORSE = {
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
}

# Factors where higher value = better (protective)
HIGHER_IS_BETTER = {
    "Balanced Diet",
}

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

# =============== PAGE CONFIG + CSS ===============

st.set_page_config(
    page_title="Cancer Risk Level Prediction",
    page_icon="🩺",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* General page */
.main {
    background-color: #f5f7fb;
    font-family: "Roboto", "Segoe UI", sans-serif;
}
h1, h2, h3 {
    font-weight: 600;
    color: #16355b;
}
.report-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.risk-low {
    color: #1b8f3a;
    font-weight: 700;
}
.risk-medium {
    color: #f0a400;
    font-weight: 700;
}
.risk-high {
    color: #c0392b;
    font-weight: 700;
}
.factor-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    margin-right: 4px;
}
.factor-badge-worse {
    background: rgba(192,57,43,0.1);
    color: #c0392b;
}
.factor-badge-better {
    background: rgba(27,143,58,0.1);
    color: #1b8f3a;
}
.small-note {
    font-size: 11px;
    color: #777;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============== SIDEBAR ===============

st.sidebar.title("Configuration")
st.sidebar.write(
    "This tool uses a trained Random Forest model to estimate a **risk level** "
    "based on patient clinical and lifestyle features."
)
st.sidebar.write(
    "**Note:** This is an academic project, not a clinical diagnostic system."
)

# =============== PAGE HEADER ===============

st.title("Cancer Risk Level Prediction Dashboard")

st.write(
    "Provide the patient's details below. The model will predict a **risk category** "
    "such as **Low**, **Medium**, or **High**, and highlight which factors are most concerning."
)

# =============== INPUT FORM ===============

with st.form("patient_form"):
    col_demo, col_lifestyle, col_symptoms = st.columns([1.1, 1.1, 1.4])

    # Demographics
    with col_demo:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])

    # Lifestyle and environment
    with col_lifestyle:
        st.subheader("Lifestyle & Exposure (1 = none, 10 = severe)")
        air_pollution = st.slider("Air Pollution", 1, 10, 5)
        alcohol_use = st.slider("Alcohol use", 1, 10, 5)
        dust_allergy = st.slider("Dust Allergy", 1, 10, 5)
        occ_hazards = st.slider("Occupational Hazards", 1, 10, 5)
        genetic_risk = st.slider("Genetic Risk", 1, 10, 5)
        balanced_diet = st.slider("Balanced Diet (higher = healthier)", 1, 10, 5)
        obesity = st.slider("Obesity", 1, 10, 5)
        smoking = st.slider("Smoking", 1, 10, 5)
        passive_smoker = st.slider("Passive Smoker", 1, 10, 5)

    # Symptoms
    with col_symptoms:
        st.subheader("Symptoms (1 = none, 10 = severe)")
        chest_pain = st.slider("Chest Pain", 1, 10, 5)
        coughing_blood = st.slider("Coughing of Blood", 1, 10, 5)
        fatigue = st.slider("Fatigue", 1, 10, 5)
        weight_loss = st.slider("Weight Loss", 1, 10, 5)
        sob = st.slider("Shortness of Breath", 1, 10, 5)
        wheezing = st.slider("Wheezing", 1, 10, 5)
        swallow_diff = st.slider("Swallowing Difficulty", 1, 10, 5)
        clubbing = st.slider("Clubbing of Finger Nails", 1, 10, 5)
        freq_cold = st.slider("Frequent Cold", 1, 10, 5)
        dry_cough = st.slider("Dry Cough", 1, 10, 5)
        snoring = st.slider("Snoring", 1, 10, 5)

    submitted = st.form_submit_button("Run Prediction")

# Build input dict matching FEATURE_COLUMNS
input_data = {
    "Age": age,
    "Gender": gender,
    "Air Pollution": air_pollution,
    "Alcohol use": alcohol_use,
    "Dust Allergy": dust_allergy,
    "OccuPational Hazards": occ_hazards,
    "Genetic Risk": genetic_risk,
    "chronic Lung Disease": 5,  # fixed mid value if not collected; adjust if you add input
    "Balanced Diet": balanced_diet,
    "Obesity": obesity,
    "Smoking": smoking,
    "Passive Smoker": passive_smoker,
    "Chest Pain": chest_pain,
    "Coughing of Blood": coughing_blood,
    "Fatigue": fatigue,
    "Weight Loss": weight_loss,
    "Shortness of Breath": sob,
    "Wheezing": wheezing,
    "Swallowing Difficulty": swallow_diff,
    "Clubbing of Finger Nails": clubbing,
    "Frequent Cold": freq_cold,
    "Dry Cough": dry_cough,
    "Snoring": snoring,
}

# Align to DF and ensure correct column order
df_input = pd.DataFrame([input_data])
df_input = df_input[FEATURE_COLUMNS]  # will error if names mismatch

# =============== PREDICTION + REPORT ===============

if submitted:
    try:
        # --- Model prediction ---
        # INPUT : df_input (1 row, all features)
        # OUTPUT: encoded class index, decoded label
        pred_encoded = model.predict(df_input)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        # If model supports predict_proba, compute class probabilities
        try:
            probs = model.predict_proba(df_input)[0]
            class_labels = label_encoder.inverse_transform(np.arange(len(probs)))
            prob_df = pd.DataFrame({
                "Level": class_labels,
                "Probability": probs
            }).sort_values("Probability", ascending=False)
        except Exception:
            prob_df = None

        # Map risk label to style
        risk_class = str(pred_label).strip().lower()
        if risk_class == "low":
            risk_style = "risk-low"
            risk_text = "Low"
        elif risk_class == "medium":
            risk_style = "risk-medium"
            risk_text = "Medium"
        else:
            risk_style = "risk-high"
            risk_text = "High"

        # --- Main result card ---
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        st.subheader("Predicted Risk Level")

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(f"<h2 class='{risk_style}'>Risk: {risk_text}</h2>", unsafe_allow_html=True)
            st.markdown(
                "<p class='small-note'>"
                "This risk level is estimated from the input factors using a Random Forest classifier. "
                "It represents relative risk within this dataset, **not a clinical diagnosis**."
                "</p>",
                unsafe_allow_html=True
            )

        with col_res2:
            if prob_df is not None:
                st.write("Class probabilities:")
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}), 
                    use_container_width=True, 
                    hide_index=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Risk scoring for factors ---
        factor_scores = []
        for feature, value in input_data.items():
            if feature not in HIGHER_IS_WORSE and feature not in HIGHER_IS_BETTER:
                continue  # skip non-scored features like Gender, Age for now

            if feature in HIGHER_IS_WORSE:
                # higher value = more risk, normalize to 0–100
                risk_score = (value - 1) / 9 * 100
                direction = "Decrease"
            else:
                # higher is protective: low values = higher risk
                risk_score = (10 - value) / 9 * 100
                direction = "Increase"

            factor_scores.append({
                "Factor": feature,
                "InputValue(1-10)": value,
                "RiskScore(0-100)": risk_score,
                "RecommendationDirection": direction
            })

        risk_df = pd.DataFrame(factor_scores).sort_values("RiskScore(0-100)", ascending=False)

        # --- Charts + Recommendations ---
        st.subheader("Risk Contribution by Factor")
        st.write(
            "Higher bars indicate features that contribute more to overall risk, "
            "based on how far they are from an ideal value."
        )

        st.bar_chart(
            data=risk_df.set_index("Factor")["RiskScore(0-100)"],
            use_container_width=True,
        )

        st.subheader("Personalized Recommendations")

        top_risky = risk_df.head(5)

        for _, row in top_risky.iterrows():
            factor = row["Factor"]
            score = row["RiskScore(0-100)"]
            direction = row["RecommendationDirection"]
            val = row["InputValue(1-10)"]

            if direction == "Decrease":
                st.markdown(
                    f"- **{factor}** — current level: **{val}/10**. "
                    f"Risk contribution ≈ **{score:.1f}/100**. Consider working to **reduce** this factor."
                )
            else:
                st.markdown(
                    f"- **{factor}** — current level: **{val}/10**. "
                    f"Risk contribution ≈ **{score:.1f}/100**. Consider working to **improve / increase** this (e.g., more balanced diet)."
                )

        st.markdown(
            "<p class='small-note'>Recommendations are based on simple rules from the input scale, "
            "not on clinical treatment guidelines.</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("Prediction failed. Check logs and ensure feature names and model inputs match.")
        st.write(e)
else:
    st.info("Fill the form and click **Run Prediction** to see the risk level and detailed report.")
