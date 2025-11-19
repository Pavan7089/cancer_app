import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import base64

# --------------------------------------
# PAGE CONFIG & CSS
# --------------------------------------
st.set_page_config(page_title="Cancer Risk Prediction", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f4f7fb; }
    .card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    .title {
        font-size: 32px;
        font-weight: 700;
        color: #1a3d6d;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #4d4d4d;
        font-size: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .risk-low { color: #1b8f3a; font-size:26px; font-weight:700; }
    .risk-medium { color: #e6a700; font-size:26px; font-weight:700; }
    .risk-high { color: #c0392b; font-size:26px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------
# LOAD ARTIFACTS
# --------------------------------------
model = joblib.load("rf_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
feature_cols = json.load(open("feature_columns.json", "r"))

# --------------------------------------
# HEADER
# --------------------------------------
st.markdown("<div class='title'>Cancer Risk Level Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill the patient details below. The model predicts Low, Medium, or High cancer risk.</div>", unsafe_allow_html=True)

# --------------------------------------
# INPUT FORM
# --------------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    inputs = {}

    # For streamlit sliders — create UI automatically
    slider_features = [
        "Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards",
        "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity",
        "Smoking", "Passive Smoker", "Chest Pain", "Coughing of Blood",
        "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
        "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold",
        "Dry Cough", "Snoring"
    ]

    # Basic Inputs
    inputs["Age"] = col1.number_input("Age", 1, 120, 45)
    gender_label = col1.selectbox("Gender", ["Male", "Female"])
    inputs["Gender"] = 1 if gender_label == "Male" else 0

    # Dynamic sliders (auto generate UI)
    all_cols = list(slider_features)
    sliders_per_col = len(all_cols) // 2

    left_features = all_cols[:len(all_cols)//2]
    right_features = all_cols[len(all_cols)//2:]

    for f in left_features:
        inputs[f] = col2.slider(f, 1, 10, 5)

    for f in right_features:
        inputs[f] = col3.slider(f, 1, 10, 5)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------
# PREDICT BUTTON
# --------------------------------------
if st.button("🔍 Predict Risk Level"):
    try:
        df = pd.DataFrame([[inputs[col] for col in feature_cols]], columns=feature_cols)

        pred_encoded = model.predict(df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0].lower()

        # --------------------------------------
        # DISPLAY RISK RESULT
        # --------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if pred_label == "low":
            st.markdown("<div class='risk-low'>Low Risk</div>", unsafe_allow_html=True)
        elif pred_label == "medium":
            st.markdown("<div class='risk-medium'>Medium Risk</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-high'>High Risk</div>", unsafe_allow_html=True)

        st.write("This prediction is based on a Random Forest model trained on patient health & lifestyle features.")

        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------
        # CLASS PROBABILITIES
        # --------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Class Probabilities")

        probs = model.predict_proba(df)[0]
        classes = label_encoder.inverse_transform(np.arange(len(probs)))

        prob_df = pd.DataFrame({"Risk Level": classes, "Probability": probs})
        prob_df["Probability"] = prob_df["Probability"].apply(lambda x: float(x))

        fig = px.bar(prob_df, x="Risk Level", y="Probability", title="Prediction Confidence",
                     text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%"))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------
        # FEATURE IMPORTANCE CHART
        # --------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Feature Importance (Model Insight)")

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig2 = px.bar(importance_df, x="Importance", y="Feature",
                      orientation="h",
                      title="Which Features Influence the Prediction Most?",
                      height=600)

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------
        # GENERATE TEXT REPORT
        # --------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📄 Downloadable Patient Risk Report")

        report_text = f"""
Cancer Risk Level Prediction Report
------------------------------------
Predicted Risk Level: {pred_label.upper()}

Input Details:
{json.dumps(inputs, indent=4)}

Model Interpretation:
- The model is based on Random Forest.
- Feature importance chart explains which factors influenced the decision.
- Probabilities show model confidence.
        """

        b64 = base64.b64encode(report_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="cancer_report.txt">📥 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed. Check model inputs & feature names.")
        st.write(e)

else:
    st.info("Fill the details above and click **Predict Risk Level**.")
