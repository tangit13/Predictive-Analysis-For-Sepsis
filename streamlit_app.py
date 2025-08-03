import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("sepsis_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Sepsis Prediction App")
st.write("Upload a CSV file with patient data for sepsis prediction.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Data")
        st.write(df)

        # Prediction
        if st.button("Predict"):
            prediction = model.predict(df)
            st.subheader("Prediction Results")
            st.write(prediction)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
