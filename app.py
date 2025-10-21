import streamlit as st
import joblib
import pandas as pd

model = joblib.load('/home/astrosanderson/Desktop/telecom-churn-prediction/models/logistic_regression_model.pkl')

st.title("Churn Prediction App (Logistic Regression)")

age = st.number_input("Customer Age", 18, 100)
income = st.number_input("Monthly Income", 0.0, 50000.0)
gender = st.selectbox("Gender", ['Male', 'Female'])
plan_type = st.selectbox("Plan Type", ['Basic', 'Premium', 'Enterprise'])

# Create input data (match training preprocessing)
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'gender_Male': [1 if gender == 'Male' else 0],
    'plan_type_Premium': [1 if plan_type == 'Premium' else 0],
    'plan_type_Enterprise': [1 if plan_type == 'Enterprise' else 0],
})

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]
    st.write(f"Prediction: {'Will Churn' if pred == 1 else 'Will Not Churn'}")
    st.write(f"Probability of Churn: {prob:.2f}")

st.markdown("---")
st.markdown("Developed by Astrosanderson")
st.markdown("[GitHub Repository]")

