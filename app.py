import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and column list
with open('advanced_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("Advanced Churn Risk Predictor")
st.write("Enter customer details to predict churn risk.")

tenure = st.slider("Customer tenure (months)", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", value=20.0)
gender = st.radio("Gender", ['Male', 'Female', 'Other'])
partner = st.radio("Partner", ['Yes', 'No'])

if st.button("Calculate Churn Risk"):
    # Create a DataFrame for the current input, ensuring column names match
    input_data = pd.DataFrame([[tenure, monthly_charges, gender, partner]], 
                              columns=['TenureMonths', 'MonthlyCharges', 'Gender', 'Partner'])
    
    # One-hot encode categorical features, similar to how the model was trained
    input_data_encoded = pd.get_dummies(input_data, columns=['Gender', 'Partner'])
    
    # Ensure all model_columns are present, filling missing with 0 (for categories not selected)
    # and reorder columns to match the training data
    features_input = pd.DataFrame(columns=model_columns)
    features_input = pd.concat([features_input, input_data_encoded], ignore_index=True)
    features_input = features_input.fillna(0) # Fill NaN from categories not present with 0
    features_input = features_input[model_columns].astype(float) # Ensure correct order and type
    
    # predict_proba returns [prob_no, prob_yes]
    probabilities = model.predict_proba(features_input)[0]
    churn_probability = probabilities[1]

    st.subheader(f"Churn Probability: {churn_probability * 100:.1f}%")

    if churn_probability > 0.5:
        st.error("High risk! This customer is likely to leave.")
    else:
        st.success("Low risk. This customer is likely to stay.")

import os
os.system('streamlit run app.py &>/content/logs.txt &')


