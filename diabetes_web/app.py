import streamlit as st
import pickle
import numpy as np

# Load model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction Web App")
st.markdown("Enter the health details below to predict diabetes risk.")

feature_names = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income']

user_inputs = []

with st.form("prediction_form"):
    for feature in feature_names:
        val = st.number_input(f"{feature}", step=1.0, format="%.2f")
        user_inputs.append(val)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array([user_inputs])
    prediction = model.predict(input_array)[0]
    result = "🟥 Diabetic" if prediction == 1 else "🟩 Not Diabetic"
    st.success(f"Prediction Result: {result}")