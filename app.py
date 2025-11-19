import streamlit as st
import pandas as pd
import joblib

# Load model + encoder
model = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.set_page_config(page_title="Cancer Level Prediction", layout="centered")
st.title("🩺 Cancer Risk Level Prediction App")
st.write("Fill the details below:")

# -----------------------
# INPUT FIELDS
# -----------------------

inputs = {}

inputs["Age"] = st.number_input("Age", 1, 120, 40)
inputs["Gender"] = st.selectbox("Gender", ["Male", "Female"])
inputs["Air Pollution"] = st.slider("Air Pollution", 1, 10, 5)
inputs["Alcohol use"] = st.slider("Alcohol use", 1, 10, 5)
inputs["Dust Allergy"] = st.slider("Dust Allergy", 1, 10, 5)
inputs["OccuPational Hazards"] = st.slider("Occupational Hazards", 1, 10, 5)
inputs["Genetic Risk"] = st.slider("Genetic Risk", 1, 10, 5)
inputs["chronic Lung Disease"] = st.slider("Chronic Lung Disease", 1, 10, 5)
inputs["Balanced Diet"] = st.slider("Balanced Diet", 1, 10, 5)
inputs["Obesity"] = st.slider("Obesity", 1, 10, 5)
inputs["Smoking"] = st.slider("Smoking", 1, 10, 5)
inputs["Passive Smoker"] = st.slider("Passive Smoker", 1, 10, 5)
inputs["Chest Pain"] = st.slider("Chest Pain", 1, 10, 5)
inputs["Coughing of Blood"] = st.slider("Coughing of Blood", 1, 10, 5)
inputs["Fatigue"] = st.slider("Fatigue", 1, 10, 5)
inputs["Weight Loss"] = st.slider("Weight Loss", 1, 10, 5)
inputs["Shortness of Breath"] = st.slider("Shortness of Breath", 1, 10, 5)
inputs["Wheezing"] = st.slider("Wheezing", 1, 10, 5)
inputs["Swallowing Difficulty"] = st.slider("Swallowing Difficulty", 1, 10, 5)
inputs["Clubbing of Finger Nails"] = st.slider("Clubbing of Finger Nails", 1, 10, 5)
inputs["Frequent Cold"] = st.slider("Frequent Cold", 1, 10, 5)
inputs["Dry Cough"] = st.slider("Dry Cough", 1, 10, 5)
inputs["Snoring"] = st.slider("Snoring", 1, 10, 5)

# Convert to dataframe
df = pd.DataFrame([inputs])

# -----------------------
# PREDICT BUTTON
# -----------------------

if st.button("Predict"):
    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"Predicted Cancer Level: **{pred_label}**")

