import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Somalia Electricity Debug Dashboard", layout="centered")

st.title("🛠 Debug: Somalia Electricity Price App")
st.markdown("This debug version helps verify if the model file is visible to Streamlit.")

# Show file list in current directory
st.subheader("📂 Current Directory Files")
st.text("\n".join(os.listdir()))

# Try loading dataset
try:
    df = pd.read_csv("somalia_electricity_data_real.csv")
    st.success("✅ Dataset loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    df = pd.DataFrame()

# Try loading model
try:
    model = joblib.load("electricity_rf_model_real.pkl")
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Model not loaded. Error: {e}")

# Only show UI if both are loaded
if not df.empty:
    st.subheader("🔮 Predict New Price")
    diesel = st.slider("Diesel Price (USD/L)", 1.0, 1.5, 1.25, 0.01)
    grid = st.slider("Grid Coverage (%)", 0, 100, 40)
    income = st.slider("Avg Household Income (USD)", 100, 500, 250)
    infra = st.slider("Infrastructure Index", 1.0, 5.0, 2.5, 0.1)

    if 'model' in locals():
        input_df = pd.DataFrame({
            'Diesel_Price_USD_per_L': [diesel],
            'Grid_Coverage_Percent': [grid],
            'Avg_Household_Income_USD': [income],
            'Infrastructure_Quality_Index': [infra]
        })
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💡 Predicted Price: ${prediction:.2f} per kWh")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
