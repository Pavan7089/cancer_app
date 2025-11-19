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

st.set_page_config(page_title="Cancer Level Prediction", layout="centered")

st.title("🩺 Cancer Risk Level Prediction")
st.write("Enter the patient details and get prediction.")

# --------------------------
# CREATE INPUT FORM
# --------------------------
inputs = {}

inputs["Age"] = st.number_input("Age", 1, 120)

gender = st.selectbox("Gender", ["Male", "Female"])
inputs["Gender"] = 1 if gender == "Male" else 0

inputs["Air Pollution"] = st.slider("Air Pollution", 1, 10, 5)
inputs["Alcohol use"] = st.slider("Alcohol use", 1, 10, 5)
inputs["Smoking"] = st.slider("Smoking", 1, 10, 5)
inputs["Coughing of Blood"] = st.slider("Coughing of Blood", 1, 10, 5)
inputs["Fatigue"] = st.slider("Fatigue", 1, 10, 5)
inputs["Shortness of Breath"] = st.slider("Shortness of Breath", 1, 10, 5)

# --------------------------
# PREDICT BUTTON
# --------------------------
if st.button("Predict Cancer Level"):
    try:
        # Ensure order of columns
        df = pd.DataFrame([[inputs[col] for col in feature_cols]], columns=feature_cols)

        pred_encoded = model.predict(df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted Cancer Level: {pred_label}")

    except Exception as e:
        st.error("Error during prediction")
        st.write(str(e))
