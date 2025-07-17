import streamlit as st
import pickle
import numpy as np
from transformers import pipeline, Conversation
from supabase import create_client, Client
import os

# --- Supabase Setup ---
SUPABASE_URL = st.secrets['https://tbyuuzmbtbwdzqgsgidc.supabase.co']
SUPABASE_KEY = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRieXV1em1idGJ3ZHpxZ3NnaWRjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI3OTEwOTgsImV4cCI6MjA2ODM2NzA5OH0.n9bHgYatFeh4lIiN_GDaduzEzdIJWELOrQt8ofe-qk8"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Authentication State ---
if "user" not in st.session_state:
    st.session_state.user = None

# --- Load Model ---
with open('diabetes.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load Chatbot Model ---
@st.cache_resource

def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

chatbot = load_chatbot()

# --- Page Config ---
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# --- Login or Signup Interface ---
if not st.session_state.user:
    auth_tab = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])

    if auth_tab == "Sign Up":
        st.sidebar.title("📝 Sign Up")
        with st.sidebar.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            signup_submit = st.form_submit_button("Sign Up")
            if signup_submit:
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.sidebar.success("✅ Sign up successful! Please verify your email.")
                except Exception as e:
                    st.sidebar.error(f"Signup failed: {e}")

    else:  # Login
        st.sidebar.title("🔐 Login")
        with st.sidebar.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")
            if login_submit:
                try:
                    user = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = user
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Login failed: {e}")

else:
    st.sidebar.success(f"👋 Welcome, {st.session_state.user.user.email}!")

    # --- Chatbot Section ---
    st.sidebar.title("🤖 Diabetes Chatbot")
    chat_input = st.sidebar.text_input("Ask a question about diabetes:")
    if chat_input:
        convo = Conversation(chat_input)
        response = chatbot(convo)
        st.sidebar.info(response.generated_responses[-1])

    # --- Main App ---
    st.title("🩺 Diabetes Prediction Web App")
    st.markdown("Enter the health details below to predict diabetes risk.")

    feature_names = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                     'PhysActivity', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth',
                     'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income']

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

    if st.button("🔓 Logout"):
        st.session_state.user = None
        st.rerun()
