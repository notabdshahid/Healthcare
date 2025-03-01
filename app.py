import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

model = joblib.load('disease_prediction_model.pkl')
mlb = joblib.load('mlb_preprocessor.pkl')

def transform_input(symptoms):
    transformed_input = mlb.transform([symptoms])
    return transformed_input

def user_input_form():
    st.title("Disease Symptom Prediction App")
    st.write("Enter your symptoms to predict the disease and the associated risk!")

    symptoms_list = ["fever", "headache", "cough", "fatigue", "skin_rash", "joint_pain", "vomiting", "diarrhea", "nausea"]

    symptoms = st.multiselect("Select Symptoms", symptoms_list)
    return symptoms

def main():
    symptoms = user_input_form()
    
    transformed_input = transform_input(symptoms)
    
    prediction = model.predict(transformed_input)
    st.write(f"Predicted Disease: {prediction[0]}")

    prob = model.predict_proba(transformed_input)[0]
    prob_df = pd.DataFrame(prob, columns=["Risk Score"], index=model.classes_).sort_values(by="Risk Score", ascending=False)
    st.write("Risk Scores (Probabilities):")
    st.write(prob_df)

if __name__ == "__main__":
    main()
