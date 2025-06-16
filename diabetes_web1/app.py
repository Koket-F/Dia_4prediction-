import streamlit as st
import pandas as pd
import pickle


# Load the model

with open('diabetes.pkl', 'rb') as f:
    final_model = pickle.load(f)

def get_user_input():
    st.title("🩺 Diabetes Risk Predictor")

    HighBP = st.radio("Do you have high blood pressure?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    HighChol = st.radio("Do you have high cholesterol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    BMI = st.number_input("Enter your Body Mass Index (BMI):", min_value=10.0, max_value=60.0, step=0.1)
    Smoker = st.radio("Have you smoked at least 100 cigarettes in your life?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Stroke = st.radio("Have you ever had a stroke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    HeartDiseaseorAttack = st.radio("Have you ever had heart disease or a heart attack?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    PhysActivity = st.radio("Have you done physical activity in the past 30 days (excluding job)?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    HvyAlcoholConsump = st.radio("Do you consume alcohol heavily? (>14 drinks/week for men, >7 for women)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    NoDocbcCost = st.radio("Were you unable to see a doctor due to cost in the past year?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    GenHlth = st.slider("General Health (1 = Excellent, 5 = Poor):", 1, 5)
    MentHlth = st.slider("Days mental health was not good in the past 30 days:", 0, 30)
    PhysHlth = st.slider("Days physical health was not good in the past 30 days:", 0, 30)
    DiffWalk = st.radio("Do you have serious difficulty walking?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Age = st.selectbox("Age Group:", options=range(1, 14), format_func=lambda x: [
        "18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
        "55–59", "60–64", "65–69", "70–74", "75–79", "80 or older"
    ][x-1])
    Education = st.selectbox("Education Level:", options=range(1, 7), format_func=lambda x: [
        "Never attended school", "Elementary", "Some high school",
        "High school graduate", "Some college", "College graduate"
    ][x-1])
    Income = st.selectbox("Income Level:", options=range(1, 9), format_func=lambda x: [
        "Less than $10,000", "$10,000–$15,000", "$15,000–$20,000",
        "$20,000–$25,000", "$25,000–$35,000", "$35,000–$50,000",
        "$50,000–$75,000", "$75,000 or more"
    ][x-1])

    features = pd.DataFrame({
        'HighBP': [HighBP],
        'HighChol': [HighChol],
        'BMI': [BMI],
        'Smoker': [Smoker],
        'Stroke': [Stroke],
        'HeartDiseaseorAttack': [HeartDiseaseorAttack],
        'PhysActivity': [PhysActivity],
        'HvyAlcoholConsump': [HvyAlcoholConsump],
        'NoDocbcCost': [NoDocbcCost],
        'GenHlth': [GenHlth],
        'MentHlth': [MentHlth],
        'PhysHlth': [PhysHlth],
        'DiffWalk': [DiffWalk],
        'Age': [Age],
        'Education': [Education],
        'Income': [Income]
    })

    return features

# Main function
def main():
    user_input = get_user_input()

    if st.button("Predict"):
        prediction = final_model.predict(user_input)

        if prediction[0] == 1:
            st.error("⚠️ Based on the data, you are likely to be diabetic. Please consult a doctor.")
        else:
            st.success("✅ Based on the data, you are not likely to be diabetic.")

if __name__ == "__main__":
    main()
