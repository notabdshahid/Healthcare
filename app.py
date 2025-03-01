import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load the trained model and preprocessor
model = joblib.load('disease_prediction_model.pkl')
mlb = joblib.load('mlb_preprocessor.pkl')

# Function to transform user input into the format required by the model
def transform_input(symptoms):
    # Transform the user symptoms into the format the model expects (e.g., MultiLabelBinarizer)
    transformed_input = mlb.transform([symptoms])
    return transformed_input

# User interface to collect symptoms
def user_input_form():
    st.title("Disease Symptom Prediction App")
    st.write("Enter your symptoms to predict the disease and the associated risk!")

    symptoms_list = ["fever", "headache", "cough", "fatigue", "skin_rash", "joint_pain", "vomiting", "diarrhea", "nausea"]

    symptoms = st.multiselect("Select Symptoms", symptoms_list)
    return symptoms

# Main function for Streamlit app
def main():
    symptoms = user_input_form()
    
    # Transform the user input to match the model's input format
    transformed_input = transform_input(symptoms)
    
    # Predict disease
    prediction = model.predict(transformed_input)
    st.write(f"Predicted Disease: {prediction[0]}")

    # Risk score (probabilities for all diseases)
    prob = model.predict_proba(transformed_input)[0]
    prob_df = pd.DataFrame(prob, columns=["Risk Score"], index=model.classes_).sort_values(by="Risk Score", ascending=False)
    st.write("Risk Scores (Probabilities):")
    st.write(prob_df)

if __name__ == "__main__":
    main()
