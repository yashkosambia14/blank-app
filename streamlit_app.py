import streamlit as st
import numpy as np

import joblib





# Load the trained model
try:
    model = joblib.load("logistic_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the app directory.")
    st.stop()  # Stop execution if the model is missing

# Function to display predictions
def display_prediction_results(model, user_input):
    prediction = model.predict(user_input)
    prob = model.predict_proba(user_input)[:, 1]
    if prediction[0] == 1:
        st.success(f"This individual is likely a LinkedIn user. Probability: {prob[0]:.2f}")
    else:
        st.warning(f"This individual is unlikely to be a LinkedIn user. Probability: {prob[0]:.2f}")

# Example Streamlit app structure
st.title("LinkedIn User Prediction App")
income = st.slider("Income (1-9)", 1, 9, step=1)
education = st.slider("Education Level (1-8)", 1, 8, step=1)
parent = st.selectbox("Parent", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
marital = st.selectbox("Marital Status", [1, 2, 3, 4, 5, 6], format_func=lambda x: [
    "Married", "Living with a partner", "Divorced", "Separated", "Widowed", "Never been married"
][x - 1])
age = st.number_input("Age (1-97)", min_value=1, max_value=97, step=1)

if st.button("Predict"):
    user_input = np.array([[income, education, parent, marital, age]])
    display_prediction_results(model, user_input)


# Function to display data definitions
def display_data_definitions():
    st.write("### Data Definitions")
    st.markdown(
        """
        - **Age**: Between 1 and 97  
        - **Education**:  
          1: Less than high school  
          2: High school incomplete  
          3: High school graduate  
          4: Some college, no degree  
          5: Two-year associate degree  
          6: Four-year college graduate  
          7: Some postgraduate or professional schooling  
          8: Postgraduate or professional degree  

        - **Parent**:  
          1: Yes  
          2: No  

        - **Marital Status**:  
          1: Married  
          2: Living with a partner  
          3: Divorced  
          4: Separated  
          5: Widowed  
          6: Never been married  

        - **Income**: 1 (lowest income) to 9 (highest income)
        """
    )

# Function to get user inputs
def get_user_inputs():
    st.write("### Provide Input Data")
    income = st.slider("Income (1-9)", 1, 9, step=1)
    education = st.slider("Education Level (1-8)", 1, 8, step=1)
    parent = st.selectbox("Parent", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    marital = st.selectbox(
        "Marital Status",
        [1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            "Married", "Living with a partner", "Divorced",
            "Separated", "Widowed", "Never been married"
        ][x - 1]
    )
    age = st.number_input("Age (1-97)", min_value=1, max_value=97, step=1)
    return income, education, parent, marital, age

# Function to display prediction results
def display_prediction_results(model, user_input):
    prediction = model.predict(user_input)
    prob = model.predict_proba(user_input)[:, 1]  # Probability for LinkedIn user

    if prediction[0] == 1:
        st.success(f"This individual is likely a LinkedIn user. Probability: {prob[0]:.2f}")
    else:
        st.warning(f"This individual is unlikely to be a LinkedIn user. Probability: {prob[0]:.2f}")

# Streamlit app
st.title("LinkedIn User Prediction App")
display_data_definitions()

# Input fields for user data
income, education, parent, marital, age = get_user_inputs()

if st.button("Predict"):
    # Prepare input for prediction
    user_input = np.array([[income, education, parent, marital, age]])

    # Assuming `model` is already loaded
    # Replace with your model loading code, e.g., joblib.load("logistic_model.pkl")
    display_prediction_results(model, user_input)
