import streamlit as st
import pandas as pd
import joblib
from utils.preprocess import preprocess_data

st.set_page_config(page_title="Sepsis Prediction", layout="wide")
st.title("🩸 Sepsis Prediction App")

model = joblib.load("sepsis_model.pkl")

option = st.sidebar.radio("Go to", ["Home", "Predict", "Model Info"])

if option == "Home":
    st.markdown("""
    ## Sepsis Prediction App  
    Upload your patient data to see sepsis risk.  
    This model uses scaled vitals (like `valuenum_scaled`) to predict sepsis.
    """)

elif option == "Predict":
    st.header("Upload Patient Data (CSV with `valuenum_scaled` column)")
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.write("Input Preview", df.head())

        try:
            X = preprocess_data(df)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]
            df['Sepsis Risk (%)'] = (probs * 100).round(2)
            df['Prediction'] = preds

            st.write("📊 Prediction Results", df)

            st.download_button("Download Results", df.to_csv(index=False), "sepsis_output.csv")

        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Model Info":
    st.markdown("""
    - **Model:** Logistic Regression  
    - **Feature:** valuenum_scaled  
    - **Threshold:** >0.6 = Sepsis  
    - Dummy model for demo purposes.
    """)
