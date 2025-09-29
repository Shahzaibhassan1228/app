import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder_model.pkl")


# Streamlit UI
st.title("🩺 Health Monitoring Prediction App")

st.write("Enter patient vital signs to predict condition")

# Collect input values
resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=20)
oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100, value=98)
o2_scale = st.selectbox("O2 Scale", [0, 1])   # e.g., 0=Normal, 1=High-flow oxygen
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=200, value=120)
heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)
temperature = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1)
on_oxygen = st.selectbox("On Oxygen", [0, 1])  # 0 = No, 1 = Yes

# Convert to DataFrame for model
input_data = pd.DataFrame({
    "Respiratory_Rate": [resp_rate],
    "Oxygen_Saturation": [oxygen_sat],
    "O2_Scale": [o2_scale],
    "Systolic_BP": [systolic_bp],
    "Heart_Rate": [heart_rate],
    "Temperature": [temperature],
    "On_Oxygen": [on_oxygen]
})


if st.button("Predict"):
    pred = model.predict(input_data)[0]
    risk_level = encoder.inverse_transform([pred])[0]  # number -> string
    st.success(f"✅ Prediction: {risk_level}")
