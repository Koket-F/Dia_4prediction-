
import streamlit as st
import pickle
import numpy as np
from supabase import create_client, Client
from transformers import pipeline

# --- Supabase config ---
SUPABASE_URL = 'https://tbyuuzmbtbwdzqgsgidc.supabase.co'
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRieXV1em1idGJ3ZHpxZ3NnaWRjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI3OTEwOTgsImV4cCI6MjA2ODM2NzA5OH0.n9bHgYatFeh4lIiN_GDaduzEzdIJWELOrQt8ofe-qk8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load diabetes prediction model ---
with open('diabetes.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load chatbot model ---
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")
chatbot = load_chatbot()

# --- User Authentication ---
def signup(email, password):
    try:
        user = supabase.auth.sign_up({
    "email": email,
    "password": password
})
        return user
    except Exception as e:
        st.error(f"Signup failed: {e}")

def login(email, password):
    try:
        user = supabase.auth.sign_in_with_password({
    "email": email,
    "password": password
})
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")

def signout():
    supabase.auth.sign_out()

# --- Save chat message to Supabase ---
def save_message(user_email, message, is_bot):
    supabase.table("chat_messages").insert({
        "user_email": user_email,
        "message": message,
        "is_bot": is_bot
    }).execute()

# --- Fetch chat history for user ---
def fetch_chat_history(user_email):
    response = supabase.table("chat_messages")\
        .select("*")\
        .eq("user_email", user_email)\
        .order("created_at", ascending=True).execute()
    if response.data:
        return response.data
    return []

# --- Streamlit App ---
st.set_page_config(page_title="Diabetes Predictor with Chatbot", layout="centered")

if "user" not in st.session_state:
    st.session_state.user = None

st.title("🩺 Diabetes Prediction Web App with Chatbot")

# --- Authentication UI ---
if st.session_state.user is None:
    auth_choice = st.sidebar.selectbox("Login or Signup?", ["Login", "Signup"])

    with st.sidebar.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(auth_choice)

    if submitted:
        if auth_choice == "Signup":
            res = signup(email, password)
            if res and res.user:
                st.success("Signed up! Please login now.")
        else:
            res = login(email, password)
            if res and res.user:
             st.session_state.user = res.user
             st.success(f"Welcome {email}!")
             st.experimental_rerun()

    st.stop()

else:
    st.sidebar.write(f"Logged in as: {st.session_state.user.email}")
    if st.sidebar.button("Logout"):
        signout()
        st.session_state.user = None
        st.experimental_rerun()

    # --- Chatbot ---
    st.sidebar.title("🤖 Diabetes Chatbot")
    chat_input = st.sidebar.text_input("Ask a question about diabetes:")
    if chat_input:
        responses = chatbot(chat_input, max_length=100, num_return_sequences=1)
        bot_reply = responses[0]['generated_text']
        st.sidebar.info(bot_reply)

        # Save messages
        save_message(st.session_state.user.email, chat_input, False)
        save_message(st.session_state.user.email, bot_reply, True)

    # Show chat history
    chat_history = fetch_chat_history(st.session_state.user.email)
    st.sidebar.markdown("---")
    st.sidebar.header("Chat History")
    for chat in chat_history[-10:]:
        speaker = "Bot" if chat["is_bot"] else "You"
        st.sidebar.write(f"{speaker}: {chat['message']}")


# --- Diabetes prediction ---
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
